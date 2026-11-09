import argparse
from pathlib import Path
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.data.dataset import load_client_dataloaders, load_pooled_train_loader
from src.baselines.centralized import train_centralized_model


def main():
    parser = argparse.ArgumentParser(description="Train centralized pooled reference U-Net model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--manifest", type=str, default="data/mock/manifest.json", help="Path to manifest.json")
    parser.add_argument("--stats", type=str, default="data/mock/norm_stats.json", help="Path to norm_stats.json")
    parser.add_argument("--max_steps", type=int, default=6000, help="Maximum optimizer steps (default: 6000)")
    parser.add_argument("--eval_interval", type=int, default=60, help="Steps between validation checks (default: 60)")
    parser.add_argument("--patience", type=int, default=15, help="Patience in validation checks")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/centralized", help="Directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # Load pooled training loader
    pooled_train = load_pooled_train_loader(
        manifest_path=args.manifest,
        stats_path=args.stats,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Load per-client val and test loaders
    client_loaders = load_client_dataloaders(
        manifest_path=args.manifest,
        stats_path=args.stats,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    val_loaders = {cid: client_loaders[cid]["val"] for cid in ["etci", "mmflood", "kuro_siwo"]}
    test_loaders = {cid: client_loaders[cid]["test"] for cid in ["etci", "mmflood", "kuro_siwo"]}

    res = train_centralized_model(
        pooled_train_loader=pooled_train,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("\n--- Centralized Test Evaluation Results ---")
    macro = res["test_results"]["macro"]
    pooled = res["test_results"]["pooled"]
    print(f"Macro Flood IoU: {macro['iou']:.4f} | Dice: {macro['dice']:.4f} | Prec: {macro['precision']:.4f} | Rec: {macro['recall']:.4f}")
    print(f"Pooled Flood IoU: {pooled['iou']:.4f} | Dice: {pooled['dice']:.4f}")
    for cid, m in res["test_results"]["per_client"].items():
        print(f"Client {cid:10s} -> IoU: {m['iou']:.4f}, Dice: {m['dice']:.4f}, Prec: {m['precision']:.4f}, Rec: {m['recall']:.4f}")


if __name__ == "__main__":
    main()
