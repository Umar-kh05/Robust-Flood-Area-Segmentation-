from pathlib import Path
import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.unet import UNet, build_unet, get_model_weights_clone
from ..metrics.flood_metrics import ConfusionMatrixAccumulator, compute_metrics_from_counts
from .client import FederatedClient


class FederatedServer:
    """
    Coordinates federated learning across clients.
    Supports FedAvg, FedProx, FedNova, and SCAFFOLD.
    Handles round orchestration, validation, early stopping, and testing.
    """

    def __init__(
        self,
        clients: list[FederatedClient],
        device: torch.device,
        seed: int = 42,
        algorithm: str = "fedavg",
        fedprox_mu: float = 0.01,
        max_rounds: int = 100,
        early_stopping_patience: int = 15,
        checkpoint_dir: str | Path = "checkpoints",
        model: nn.Module | None = None,
    ):
        self.clients = clients
        self.device = device
        self.seed = seed
        self.algorithm = algorithm.lower()
        self.fedprox_mu = fedprox_mu
        self.max_rounds = max_rounds
        self.patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build or use provided initial global model
        if model is not None:
            self.global_model = model.to(device)
        else:
            self.global_model = build_unet(seed=seed).to(device)

        # Worker model reused across client updates to avoid repeated deepcopies
        self.worker_model = copy.deepcopy(self.global_model)

        # Compute static aggregation weights based on training tile counts
        total_tiles = sum(c.num_train_tiles for c in clients)
        self.weights = {c.client_id: c.num_train_tiles / total_tiles for c in clients}

        # Initialize SCAFFOLD global control variates if needed
        self.c_global: dict[str, torch.Tensor] | None = None
        if self.algorithm == "scaffold":
            self.c_global = {
                name: torch.zeros_like(param.data)
                for name, param in self.global_model.named_parameters()
                if param.requires_grad
            }
            for c in self.clients:
                c.init_scaffold_control(self.global_model)

        self.history = []

    def evaluate_model(
        self, model: nn.Module, loader: DataLoader
    ) -> dict[str, float]:
        """Evaluates segmentation performance on given dataloader."""
        model.eval()
        accumulator = ConfusionMatrixAccumulator(threshold=0.5, ignore_index=255)

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                probs = model.predict_proba(images)
                accumulator.update(probs, masks)

        return accumulator.compute()

    def aggregate_updates(
        self, client_results: list[dict], global_w: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Performs algorithm-specific server aggregation."""
        new_state = {}

        if self.algorithm in ["fedavg", "fedprox"]:
            for key in global_w.keys():
                weighted_sum = sum(
                    self.weights[res["client_id"]] * res["state_dict"][key]
                    for res in client_results
                )
                new_state[key] = weighted_sum

        elif self.algorithm == "fednova":
            # FedNova: tau = sum(p_i * a_i), update = w - tau * sum(p_i * d_i)
            tau = sum(self.weights[res["client_id"]] * res["num_steps"] for res in client_results)
            for key in global_w.keys():
                if key in client_results[0]["delta_w"]:
                    accum_d = torch.zeros_like(global_w[key])
                    for res in client_results:
                        p_i = self.weights[res["client_id"]]
                        a_i = res["num_steps"]
                        delta_w = res["delta_w"][key]
                        accum_d += p_i * (delta_w / a_i)
                    new_state[key] = global_w[key] + tau * accum_d
                else:
                    new_state[key] = global_w[key]

        elif self.algorithm == "scaffold":
            for key in global_w.keys():
                if key in client_results[0]["delta_w"]:
                    weighted_delta = sum(
                        self.weights[res["client_id"]] * res["delta_w"][key]
                        for res in client_results
                    )
                    new_state[key] = global_w[key] + weighted_delta
                else:
                    new_state[key] = global_w[key]

            # Update global control variates
            for name in self.c_global.keys():
                delta_c_agg = sum(
                    self.weights[res["client_id"]] * res["delta_c"][name]
                    for res in client_results
                )
                self.c_global[name] += delta_c_agg.to(self.device)

        return new_state

    def run_training(self, step_budget_per_client: dict[str, int]) -> dict:
        """
        Runs federated training up to max_rounds.
        Tracks validation IoU on all clients, checkpointing best macro-mean IoU.
        """
        best_mean_val_iou = -1.0
        best_round = -1
        patience_counter = 0
        best_checkpoint_path = self.checkpoint_dir / f"{self.algorithm}_seed{self.seed}_best.pt"

        print(
            f"Starting {self.algorithm.upper()} training (Seed: {self.seed}, Max rounds: {self.max_rounds})",
            flush=True,
        )

        for round_idx in range(1, self.max_rounds + 1):
            client_results = []
            global_state = get_model_weights_clone(self.global_model)

            # Local training on each participating client
            for client in self.clients:
                self.worker_model.load_state_dict(self.global_model.state_dict())
                steps = step_budget_per_client.get(client.client_id, 20)

                res = client.train_round(
                    model=self.worker_model,
                    num_steps=steps,
                    algorithm=self.algorithm,
                    fedprox_mu=self.fedprox_mu,
                    server_control=self.c_global,
                )
                client_results.append(res)

            # Server aggregation
            updated_state = self.aggregate_updates(client_results, global_state)
            self.global_model.load_state_dict(updated_state)


            # Client validation evaluation
            val_metrics = {}
            val_ious = []
            for client in self.clients:
                metrics = self.evaluate_model(self.global_model, client.val_loader)
                val_metrics[client.client_id] = metrics
                val_ious.append(metrics["iou"])

            mean_val_iou = sum(val_ious) / len(val_ious)

            round_log = {
                "round": round_idx,
                "mean_val_iou": mean_val_iou,
                "client_val_ious": {c.client_id: m["iou"] for c, m in zip(self.clients, val_metrics.values())},
                "client_train_losses": {r["client_id"]: r["train_loss"] for r in client_results},
            }
            self.history.append(round_log)

            print(
                f"Round {round_idx:03d} | Mean Val IoU: {mean_val_iou:.4f} | "
                + " | ".join(f"{c}: {m['iou']:.4f}" for c, m in val_metrics.items()),
                flush=True,
            )

            # Checkpoint and early stopping rule
            if mean_val_iou > best_mean_val_iou:
                best_mean_val_iou = mean_val_iou
                best_round = round_idx
                patience_counter = 0
                torch.save(
                    {
                        "round": round_idx,
                        "state_dict": self.global_model.state_dict(),
                        "mean_val_iou": mean_val_iou,
                        "val_metrics": val_metrics,
                        "algorithm": self.algorithm,
                        "seed": self.seed,
                    },
                    best_checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at round {round_idx} (no improvement for {self.patience} rounds)")
                    break

        print(f"Selected best checkpoint from round {best_round} with Mean Val IoU: {best_mean_val_iou:.4f}")

        # Final evaluation on untouched test events
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        self.global_model.load_state_dict(checkpoint["state_dict"])
        test_results = self.evaluate_test_sets()

        summary = {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "best_round": best_round,
            "best_val_iou": best_mean_val_iou,
            "test_results": test_results,
            "history": self.history,
        }

        # Save run summary
        summary_path = self.checkpoint_dir / f"{self.algorithm}_seed{self.seed}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def evaluate_test_sets(self) -> dict:
        """Evaluates selected best checkpoint across all client test sets."""
        per_client = {}
        pooled_tp = 0
        pooled_fp = 0
        pooled_fn = 0
        pooled_tn = 0

        for client in self.clients:
            m = self.evaluate_model(self.global_model, client.test_loader)
            per_client[client.client_id] = m
            pooled_tp += m["tp"]
            pooled_fp += m["fp"]
            pooled_fn += m["fn"]
            pooled_tn += m["tn"]

        # Macro client results (arithmetic mean across clients)
        macro_iou = sum(m["iou"] for m in per_client.values()) / len(per_client)
        macro_dice = sum(m["dice"] for m in per_client.values()) / len(per_client)
        macro_precision = sum(m["precision"] for m in per_client.values()) / len(per_client)
        macro_recall = sum(m["recall"] for m in per_client.values()) / len(per_client)

        # Pooled metrics
        pooled = compute_metrics_from_counts(pooled_tp, pooled_fp, pooled_fn, pooled_tn)

        return {
            "per_client": per_client,
            "macro": {
                "iou": macro_iou,
                "dice": macro_dice,
                "precision": macro_precision,
                "recall": macro_recall,
            },
            "pooled": pooled,
        }
