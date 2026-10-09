import argparse
import json
from pathlib import Path
import numpy as np


def compute_normalization_percentiles(
    train_tiles: list[dict],
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Computes 1st and 99th percentiles of valid pixels in training events only,
    separately for each client and channel (VV, VH).
    """
    client_values = {"etci": {"vv": [], "vh": []}, "mmflood": {"vv": [], "vh": []}, "kuro_siwo": {"vv": [], "vh": []}}

    for item in train_tiles:
        client = item["client"]
        if client not in client_values:
            continue

        file_path = Path(item["path"])
        if not file_path.exists():
            continue

        data = np.load(file_path)
        img = data["image"]  # (2, H, W)
        mask = data["mask"]  # (H, W)

        valid = (mask != 255) & np.isfinite(img[0]) & np.isfinite(img[1])
        if not valid.any():
            continue

        # Subsample for memory efficiency if tile is large
        vv_valid = img[0][valid]
        vh_valid = img[1][valid]

        if len(vv_valid) > 2000:
            sub_idx = np.random.choice(len(vv_valid), size=2000, replace=False)
            vv_valid = vv_valid[sub_idx]
            vh_valid = vh_valid[sub_idx]

        client_values[client]["vv"].extend(vv_valid.tolist())
        client_values[client]["vh"].extend(vh_valid.tolist())

    stats = {}
    for client, channels in client_values.items():
        stats[client] = {}
        for ch in ["vv", "vh"]:
            arr = np.array(channels[ch], dtype=np.float32)
            if len(arr) > 0:
                p1 = float(np.percentile(arr, 1))
                p99 = float(np.percentile(arr, 99))
            else:
                p1, p99 = -25.0, 0.0  # sensible default SAR dB range
            stats[client][ch] = (p1, p99)

    return stats


def generate_mock_dataset(base_dir: Path, tiles_per_split: int = 12):
    """
    Generates a lightweight mock dataset representing ETCI, MMFlood, and Kuro Siwo
    to allow full end-to-end testing of data loaders, model, federated algorithms,
    and evaluation routines.
    """
    tiles_dir = base_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    manifest_items = []
    clients = ["etci", "mmflood", "kuro_siwo"]

    client_events = {
        "etci": {
            "train": ["nebraska_flood_01", "nebraska_flood_02"],
            "val": ["northal_event_01"],
            "test": ["bangladesh_monsoon_01"],
        },
        "mmflood": {
            "train": ["EMS101", "EMS102", "EMS103"],
            "val": ["EMS201"],
            "test": ["EMS301"],
        },
        "kuro_siwo": {
            "train": ["kuro_train_evt01", "kuro_train_evt02"],
            "val": ["kuro_val_evt01"],
            "test": ["kuro_test_holdout01"],
        },
    }

    np.random.seed(42)

    for client in clients:
        for split, events in client_events[client].items():
            for i in range(tiles_per_split):
                event_id = events[i % len(events)]
                tile_id = f"{client}_{split}_{i:03d}"
                tile_path = tiles_dir / f"{tile_id}.npz"

                # Synthetic SAR channels (dB-like range: VV roughly [-25, 0], VH roughly [-30, -5])
                vv = np.random.normal(loc=-12.0, scale=4.0, size=(256, 256)).astype(np.float32)
                vh = np.random.normal(loc=-18.0, scale=4.0, size=(256, 256)).astype(np.float32)
                image = np.stack([vv, vh], axis=0)

                # Synthetic mask: 0 (non-flood), 1 (flood), 255 (ignore/invalid)
                mask = np.zeros((256, 256), dtype=np.uint8)

                # Add some simulated water body / flooded patch
                if i % 3 != 0:  # 2/3 of tiles contain some flood
                    cx, cy = np.random.randint(50, 206, size=2)
                    r = np.random.randint(20, 60)
                    yy, xx = np.ogrid[:256, :256]
                    flood_circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
                    mask[flood_circle] = 1

                # Random small invalid border or mask patch
                if i % 4 == 0:
                    mask[:8, :] = 255
                    mask[:, :8] = 255

                np.savez_compressed(tile_path, image=image, mask=mask)

                valid_px = int((mask != 255).sum())
                flood_px = int((mask == 1).sum())

                manifest_items.append(
                    {
                        "tile_id": tile_id,
                        "client": client,
                        "split": split,
                        "event_id": event_id,
                        "path": str(tile_path),
                        "valid_pixels": valid_px,
                        "flood_pixels": flood_px,
                    }
                )

    # Save manifest
    manifest_path = base_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_items, f, indent=2)

    # Compute training percentiles
    train_items = [it for it in manifest_items if it["split"] == "train"]
    stats = compute_normalization_percentiles(train_items)

    stats_path = base_dir / "norm_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Generated mock dataset with {len(manifest_items)} tiles at {base_dir}")
    print(f"Saved manifest to {manifest_path} and stats to {stats_path}")
    return manifest_items, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and audit dataset")
    parser.add_argument("--mock", action="store_true", help="Generate lightweight mock dataset for testing")
    parser.add_argument("--output_dir", type=str, default="data/mock", help="Output directory")
    args = parser.parse_args()

    if args.mock:
        generate_mock_dataset(Path(args.output_dir))
