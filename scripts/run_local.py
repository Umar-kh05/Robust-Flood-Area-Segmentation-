import argparse
from pathlib import Path
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.data.dataset import load_client_dataloaders
from src.baselines.local import train_local_model


def main():
    parser = argparse.ArgumentParser(description="Train local-only reference U-Net models")
    parser.add_argument(
        "--client",
        type=str,
        default="all",
        choices=["all", "etci", "mmflood", "kuro_siwo"],
        help="Client to train local model for",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--manifest", type=str, default="data/mock/manifest.json", help="Path to manifest.json")
    parser.add_argument("--stats", type=str, default="data/mock/norm_stats.json", help="Path to norm_stats.json")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum local optimizer steps (default: 2000)")
    parser.add_argument("--eval_interval", type=int, default=20, help="Steps between validation checks (default: 20)")
    parser.add_argument("--patience", type=int, default=15, help="Patience in validation checks")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/local", help="Directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    client_loaders = load_client_dataloaders(
        manifest_path=args.manifest,
        stats_path=args.stats,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    targets = ["etci", "mmflood", "kuro_siwo"] if args.client == "all" else [args.client]

    for cid in targets:
        loaders = client_loaders[cid]
        print(f"\n=================== Training Local Model for {cid.upper()} ===================")
        res = train_local_model(
            client_id=cid,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
            device=device,
            seed=args.seed,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            patience=args.patience,
            checkpoint_dir=args.checkpoint_dir,
        )
        tm = res["test_metrics"]
        print(f"Results for {cid}: Best Step: {res['best_step']}, Val IoU: {res['best_val_iou']:.4f}")
        print(f"Test IoU: {tm['iou']:.4f}, Dice: {tm['dice']:.4f}, Prec: {tm['precision']:.4f}, Rec: {tm['recall']:.4f}")


if __name__ == "__main__":
    main()
