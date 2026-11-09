import math
import numpy as np


def decode_kuro_siwo_target(mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Kuro Siwo label mapping:
      class 2 (flood) -> 1
      class 0 (no water), class 1 (permanent water) -> 0
      valid_mask == 0 -> 255 (ignore)
    """
    target = np.full(mask.shape, 255, dtype=np.uint8)

    is_valid = valid_mask == 1
    # 0 (no water) and 1 (permanent water) -> 0
    non_flood = is_valid & ((mask == 0) | (mask == 1))
    target[non_flood] = 0

    # 2 (flood) -> 1
    flood = is_valid & (mask == 2)
    target[flood] = 1

    return target


def split_kuro_siwo_events(
    train_event_ids: list[str], test_event_ids: list[str], seed: int = 42
) -> dict[str, list[str]]:
    """
    Preserves author's test events as untouched test set.
    Splits train events into train (80%) and validation (20%, at least 1) using default_rng(seed).
    """
    sorted_train_events = sorted(set(train_event_ids))
    n = len(sorted_train_events)
    if n < 2:
        raise ValueError(f"Need at least 2 training events to split Kuro Siwo, got {n}")

    rng = np.random.default_rng(seed)
    shuffled = sorted_train_events.copy()
    rng.shuffle(shuffled)

    val_count = max(1, math.floor(0.2 * n))
    val_events = shuffled[:val_count]
    train_events = shuffled[val_count:]

    return {
        "train": sorted(train_events),
        "val": sorted(val_events),
        "test": sorted(set(test_event_ids)),
    }
