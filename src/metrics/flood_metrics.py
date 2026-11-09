import math
import torch


def compute_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Computes segmentation metrics from accumulated pixel counts."""
    iou_denom = tp + fp + fn
    iou = float(tp) / iou_denom if iou_denom > 0 else float("nan")

    dice_denom = 2 * tp + fp + fn
    dice = (2.0 * tp) / dice_denom if dice_denom > 0 else float("nan")

    prec_denom = tp + fp
    precision = float(tp) / prec_denom if prec_denom > 0 else float("nan")

    rec_denom = tp + fn
    recall = float(tp) / rec_denom if rec_denom > 0 else float("nan")

    total_pixels = tp + fp + fn + tn
    accuracy = float(tp + tn) / total_pixels if total_pixels > 0 else float("nan")

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class ConfusionMatrixAccumulator:
    """Accumulates TP, FP, FN, TN over valid evaluation pixels for a client."""

    def __init__(self, threshold: float = 0.5, ignore_index: int = 255):
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        if probs.dim() == 4 and probs.size(1) == 1:
            probs = probs.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return

        valid_p = probs[valid_mask]
        valid_y = targets[valid_mask].long()

        pred_binary = (valid_p >= self.threshold).long()

        tp = ((pred_binary == 1) & (valid_y == 1)).sum().item()
        fp = ((pred_binary == 1) & (valid_y == 0)).sum().item()
        fn = ((pred_binary == 0) & (valid_y == 1)).sum().item()
        tn = ((pred_binary == 0) & (valid_y == 0)).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def compute(self) -> dict[str, float]:
        return compute_metrics_from_counts(self.tp, self.fp, self.fn, self.tn)
