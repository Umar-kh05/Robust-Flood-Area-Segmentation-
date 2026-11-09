import math
import numpy as np


def decode_mmflood_target(
    flood_mask: np.ndarray,
    hydro_mask: np.ndarray,
    sar_vv: np.ndarray,
    sar_vh: np.ndarray,
) -> np.ndarray:
    """
    Implements Equation 1 from paper:
      y = 255 if V=0 or (F=1 and H=1)  (conflict ignored)
      y = 1   if V=1 and F=1 and H=0
      y = 0   if V=1 and F=0
    where:
      V=0 if non-finite SAR, no-data, or mask==255
    """
    valid_sar = np.isfinite(sar_vv) & np.isfinite(sar_vh)
    valid_flood = (flood_mask != 255) & ((flood_mask == 0) | (flood_mask == 1))
    v = valid_sar & valid_flood

    f = (flood_mask == 1).astype(np.uint8)
    h = ((hydro_mask == 1) | (hydro_mask == 255)).astype(np.uint8)

    target = np.full(flood_mask.shape, 255, dtype=np.uint8)

    # Condition: V = 1 and F = 0 -> 0
    cond_0 = v & (f == 0)
    target[cond_0] = 0

    # Condition: V = 1 and F = 1 and H = 0 -> 1
    cond_1 = v & (f == 1) & (h == 0)
    target[cond_1] = 1

    # Conflict: V = 1 and F = 1 and H = 1 -> 255 (already 255)
    return target


def split_mmflood_events(
    event_ids: list[str], seed: int = 42
) -> dict[str, list[str]]:
    """
    Deterministic 60/20/20 group-level split by EMS activation code.
    val = max(1, floor(0.2 * G))
    test = max(1, floor(0.2 * G))
    train = remainder
    """
    sorted_events = sorted(set(event_ids))
    g = len(sorted_events)
    if g < 3:
        raise ValueError(f"Need at least 3 events to split MMFlood, got {g}")

    rng = np.random.default_rng(seed)
    shuffled = sorted_events.copy()
    rng.shuffle(shuffled)

    val_count = max(1, math.floor(0.2 * g))
    test_count = max(1, math.floor(0.2 * g))

    test_events = shuffled[:test_count]
    val_events = shuffled[test_count : test_count + val_count]
    train_events = shuffled[test_count + val_count :]

    return {
        "train": sorted(train_events),
        "val": sorted(val_events),
        "test": sorted(test_events),
    }
