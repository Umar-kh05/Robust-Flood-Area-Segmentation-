"""
Build a reduced, Kaggle-ready federated dataset from data/federated_ready.

For each client (etci, kurosiwo, mmflood) we keep exactly 200 tiles:
    - 100 flood-positive tiles (flood pixels > 200) when available
    - 100 non-flood / negative tiles
The 200 tiles are then shuffled with a fixed seed and split into
train / val / test = 140 / 30 / 30 (70% / 15% / 15%).

Output layout (identical to what the training notebook expects):
    data/federated_kaggle/<client>/<split>/images/<name>.npy
    data/federated_kaggle/<client>/<split>/masks/<name>.npy
"""
import shutil
from pathlib import Path

import numpy as np

SRC = Path("data/federated_ready")
DST = Path("data/federated_kaggle")
CLIENTS = ["etci", "kurosiwo", "mmflood"]

N_PER_CLIENT = 200
N_POS = 100
N_NEG = 100
SPLIT = {"train": 140, "val": 30, "test": 30}
POS_THRESHOLD_PX = 200
SEED = 42


def collect(client):
    """Return (positives, negatives) lists of (img_path, mask_path) tuples."""
    pos, neg = [], []
    for split in ["train", "val", "test"]:
        img_dir = SRC / client / split / "images"
        msk_dir = SRC / client / split / "masks"
        for img in sorted(img_dir.glob("*.npy")):
            msk = msk_dir / img.name
            if not msk.exists():
                continue
            m = np.load(msk)
            if (m == 1).sum() > POS_THRESHOLD_PX:
                pos.append((img, msk))
            else:
                neg.append((img, msk))
    return pos, neg


def main():
    rng = np.random.default_rng(SEED)
    if DST.exists():
        shutil.rmtree(DST)

    summary = {}
    for client in CLIENTS:
        pos, neg = collect(client)
        rng.shuffle(pos)
        rng.shuffle(neg)
        take_pos = pos[:min(N_POS, len(pos))]
        n_neg_needed = N_PER_CLIENT - len(take_pos)
        take_neg = neg[:min(N_NEG, n_neg_needed, len(neg))]
        chosen = take_pos + take_neg
        rng.shuffle(chosen)  # interleave positives across all splits

        # split sequentially
        idx = 0
        counts = {}
        for split, n in SPLIT.items():
            batch = chosen[idx: idx + n]
            idx += n
            counts[split] = len(batch)
            for img, msk in batch:
                (DST / client / split / "images").mkdir(parents=True, exist_ok=True)
                (DST / client / split / "masks").mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, DST / client / split / "images" / img.name)
                shutil.copy2(msk, DST / client / split / "masks" / msk.name)
        summary[client] = {
            "total": len(chosen),
            "pos_kept": len(take_pos),
            "neg_kept": len(take_neg),
            **counts,
        }

    print("Built reduced dataset at", DST)
    for c, s in summary.items():
        print(f"  {c:9s}: total={s['total']} pos={s['pos_kept']} neg={s['neg_kept']} "
              f"train={s['train']} val={s['val']} test={s['test']}")


if __name__ == "__main__":
    main()
