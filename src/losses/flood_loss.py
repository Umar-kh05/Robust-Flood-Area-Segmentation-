import torch
import torch.nn as nn
import torch.nn.functional as F


class FloodLoss(nn.Module):
    """
    Combined masked Binary Cross-Entropy and masked Soft Dice loss.
    Calculated across valid pixels (target != 255) over the entire mini-batch.
    If no positive target pixels exist in the batch, Dice term is omitted.
    """

    def __init__(self, eps: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        valid_mask = targets != self.ignore_index
        valid_count = valid_mask.sum().item()

        # All-invalid batch: caller should discard and fetch next batch
        if valid_count == 0:
            return None, {"loss": 0.0, "bce": 0.0, "dice": 0.0, "valid_pixels": 0}

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask].float()

        bce = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction="mean")

        num_positives = valid_targets.sum().item()
        if num_positives == 0:
            total_loss = bce
            dice_val = 0.0
        else:
            p = torch.sigmoid(valid_logits)
            intersection = torch.sum(p * valid_targets)
            cardinality = torch.sum(p) + torch.sum(valid_targets)
            dice = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)
            total_loss = 0.5 * bce + 0.5 * dice
            dice_val = dice.item()

        metrics = {
            "loss": total_loss.item(),
            "bce": bce.item(),
            "dice": dice_val,
            "valid_pixels": valid_count,
        }
        return total_loss, metrics
