import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.losses.flood_loss import FloodLoss


def test_flood_loss_regular():
    criterion = FloodLoss()
    logits = torch.randn(2, 1, 64, 64)
    # Mixed 0, 1, and 255 (ignore)
    targets = torch.zeros(2, 1, 64, 64, dtype=torch.long)
    targets[:, :, 10:20, 10:20] = 1
    targets[:, :, :5, :5] = 255

    loss, meta = criterion(logits, targets)
    assert loss is not None
    assert loss.item() > 0.0
    assert meta["dice"] > 0.0
    assert meta["bce"] > 0.0


def test_flood_loss_no_positives():
    criterion = FloodLoss()
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.zeros(2, 1, 64, 64, dtype=torch.long)
    targets[:, :, :5, :5] = 255  # only 0s and 255s

    loss, meta = criterion(logits, targets)
    assert loss is not None
    assert meta["dice"] == 0.0, "Dice should be omitted when no positive target pixels exist"
    assert meta["loss"] == meta["bce"]


def test_flood_loss_all_invalid():
    criterion = FloodLoss()
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.full((2, 1, 64, 64), 255, dtype=torch.long)

    loss, meta = criterion(logits, targets)
    assert loss is None, "All-invalid batch must return None loss"
    assert meta["valid_pixels"] == 0


if __name__ == "__main__":
    test_flood_loss_regular()
    test_flood_loss_no_positives()
    test_flood_loss_all_invalid()
    print("test_loss passed!")
