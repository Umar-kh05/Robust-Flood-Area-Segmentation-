import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..losses.flood_loss import FloodLoss
from ..data.dataset import StepBasedBatchSampler


class FederatedClient:
    """
    Represents an individual client in federated training.
    Executes local optimization steps on local training data.
    """

    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_train_tiles: int,
        device: torch.device,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        grad_norm_limit: float = 5.0,
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_train_tiles = num_train_tiles
        self.device = device

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.grad_norm_limit = grad_norm_limit

        self.sampler = StepBasedBatchSampler(train_loader)
        self.criterion = FloodLoss()

        # SCAFFOLD local control variate
        self.c_local: dict[str, torch.Tensor] | None = None

    def init_scaffold_control(self, model: nn.Module):
        """Initializes client control variate with zeros matching model parameters."""
        self.c_local = {
            name: torch.zeros_like(param.data)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def train_round(
        self,
        model: nn.Module,
        num_steps: int,
        algorithm: str = "fedavg",
        fedprox_mu: float = 0.01,
        server_control: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        """
        Runs local training updates for num_steps mini-batches.
        Returns updated model state dict and algorithm-specific update metadata.
        """
        model.to(self.device)
        model.train()

        # Keep snapshot of global weights received
        global_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=0.0,
            weight_decay=self.weight_decay,
        )

        completed_steps = 0
        total_loss_accum = 0.0

        while completed_steps < num_steps:
            batch = self.sampler.next_batch()
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            optimizer.zero_grad()
            logits = model(images)
            loss, meta = self.criterion(logits, masks)

            # Skip all-invalid batches
            if loss is None:
                continue

            # FedProx proximal term
            if algorithm == "fedprox" and fedprox_mu > 0:
                prox_term = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        prox_term += torch.sum((param - global_params[name]) ** 2)
                loss = loss + 0.5 * fedprox_mu * prox_term

            loss.backward()

            # SCAFFOLD gradient correction: g <- g - c_i + c
            if algorithm == "scaffold" and self.c_local is not None and server_control is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            c_i = self.c_local[name].to(self.device)
                            c_s = server_control[name].to(self.device)
                            param.grad.add_(c_s - c_i)

            if self.grad_norm_limit > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_limit)

            optimizer.step()

            completed_steps += 1
            total_loss_accum += loss.item()

        # Calculate update payloads
        updated_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

        delta_w = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                delta_w[name] = (param.detach() - global_params[name]).cpu()

        result = {
            "client_id": self.client_id,
            "state_dict": updated_state,
            "delta_w": delta_w,
            "num_steps": completed_steps,
            "train_loss": total_loss_accum / max(completed_steps, 1),
            "num_tiles": self.num_train_tiles,
        }

        # SCAFFOLD control variate update
        if algorithm == "scaffold" and self.c_local is not None and server_control is not None:
            c_new = {}
            delta_c = {}
            step_factor = 1.0 / (completed_steps * self.lr)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    c_i = self.c_local[name]
                    c_s = server_control[name].cpu()
                    w_diff = (global_params[name].cpu() - param.detach().cpu())
                    # c_i^+ = c_i - c_s + (w_t - w_{t+1}) / (K * eta)
                    c_next = c_i - c_s + step_factor * w_diff
                    delta_c[name] = c_next - c_i
                    c_new[name] = c_next

            self.c_local = c_new
            result["delta_c"] = delta_c

        return result
