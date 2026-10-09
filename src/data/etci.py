from pathlib import Path
import numpy as np
from PIL import Image


def decode_etci_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Decodes ETCI 2021 mask:
      - 255 -> 1 (flood)
      - 0 -> 0 (non-flood)
      - Masks in {0, 1} are accepted directly
      - Unresolved or invalid values become 255 (ignore)
    """
    unique_vals = set(np.unique(raw_mask))
    target = np.full_like(raw_mask, 255, dtype=np.uint8)

    if unique_vals.issubset({0, 255}):
        target[raw_mask == 0] = 0
        target[raw_mask == 255] = 1
    elif unique_vals.issubset({0, 1}):
        target[raw_mask == 0] = 0
        target[raw_mask == 1] = 1
    else:
        # Standard known mapping for mixed/extended formats
        valid_0 = raw_mask == 0
        valid_flood = (raw_mask == 255) | (raw_mask == 1)
        target[valid_0] = 0
        target[valid_flood] = 1

    return target


def get_etci_split_assignment(region_name: str) -> str:
    """
    Paper split assignment:
      Nebraska -> train
      North Alabama (northal) -> val
      Bangladesh -> test
    """
    r_lower = region_name.lower()
    if "nebraska" in r_lower:
        return "train"
    elif "northal" in r_lower or "alabama" in r_lower:
        return "val"
    elif "bangladesh" in r_lower:
        return "test"
    else:
        raise ValueError(f"Unknown ETCI region group: {region_name}")
