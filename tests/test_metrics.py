import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.metrics.flood_metrics import ConfusionMatrixAccumulator, compute_metrics_from_counts


def test_metrics_calculation():
    # 10 TP, 2 FP, 3 FN, 100 TN
    m = compute_metrics_from_counts(tp=10, fp=2, fn=3, tn=100)
    assert round(m["iou"], 4) == round(10 / (10 + 2 + 3), 4)
    assert round(m["dice"], 4) == round(20 / (20 + 2 + 3), 4)
    assert round(m["precision"], 4) == round(10 / 12, 4)
    assert round(m["recall"], 4) == round(10 / 13, 4)


def test_accumulator_with_ignore():
    accum = ConfusionMatrixAccumulator(threshold=0.5, ignore_index=255)
    probs = torch.tensor([[[[0.8, 0.2], [0.1, 0.9]]]])
    targets = torch.tensor([[[[1, 0], [255, 1]]]])

    accum.update(probs, targets)
    res = accum.compute()

    # Pixel (0,0): prob 0.8 >= 0.5 (pred=1), target=1 -> TP
    # Pixel (0,1): prob 0.2 < 0.5 (pred=0), target=0 -> TN
    # Pixel (1,0): target 255 -> ignored!
    # Pixel (1,1): prob 0.9 >= 0.5 (pred=1), target=1 -> TP
    assert res["tp"] == 2
    assert res["tn"] == 1
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["iou"] == 1.0


if __name__ == "__main__":
    test_metrics_calculation()
    test_accumulator_with_ignore()
    print("test_metrics passed!")
