import argparse
from pathlib import Path
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.unet import UNet
from src.data.dataset import load_client_dataloaders
from src.metrics.flood_metrics import ConfusionMatrixAccumulator, compute_metrics_from_counts


def evaluate_client(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    accum = ConfusionMatrixAccumulator(threshold=0.5, ignore_index=255)
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            probs = model.predict_proba(imgs)
            accum.update(probs, masks)
    return accum.compute()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint on test events")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--manifest", type=str, default="data/mock/manifest.json", help="Path to manifest.json")
    parser.add_argument("--stats", type=str, default="data/mock/norm_stats.json", help="Path to norm_stats.json")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint from: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    model = UNet(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(state_dict)

    client_loaders = load_client_dataloaders(
        manifest_path=args.manifest,
        stats_path=args.stats,
        batch_size=args.batch_size,
    )

    per_client = {}
    p_tp, p_fp, p_fn, p_tn = 0, 0, 0, 0

    print("\n--- Evaluation on Test Partitions ---")
    for cid in ["etci", "mmflood", "kuro_siwo"]:
        loader = client_loaders[cid]["test"]
        metrics = evaluate_client(model, loader, device)
        per_client[cid] = metrics
        p_tp += metrics["tp"]
        p_fp += metrics["fp"]
        p_fn += metrics["fn"]
        p_tn += metrics["tn"]
        print(f"Client {cid:10s} -> IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}")

    macro_iou = sum(m["iou"] for m in per_client.values()) / len(per_client)
    macro_dice = sum(m["dice"] for m in per_client.values()) / len(per_client)
    macro_prec = sum(m["precision"] for m in per_client.values()) / len(per_client)
    macro_rec = sum(m["recall"] for m in per_client.values()) / len(per_client)

    pooled = compute_metrics_from_counts(p_tp, p_fp, p_fn, p_tn)

    print("\n--- Summary Metrics ---")
    print(f"Macro:  IoU: {macro_iou:.4f} | Dice: {macro_dice:.4f} | Prec: {macro_prec:.4f} | Rec: {macro_rec:.4f}")
    print(f"Pooled: IoU: {pooled['iou']:.4f} | Dice: {pooled['dice']:.4f} | Prec: {pooled['precision']:.4f} | Rec: {pooled['recall']:.4f}")


if __name__ == "__main__":
    main()
