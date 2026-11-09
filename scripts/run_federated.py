import argparse
from pathlib import Path
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.data.dataset import load_client_dataloaders
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer


def get_step_budgets(experiment: str, seed: int) -> dict[str, int]:
    """
    Returns client local step budget per round.
    Main experiment: 20 steps each.
    Unequal experiment: rotates [10, 20, 30] across seeds 42, 43, 44.
    """
    if experiment == "main":
        return {"etci": 20, "mmflood": 20, "kuro_siwo": 20}

    # Unequal computation schedule
    rotations = {
        42: {"etci": 10, "mmflood": 20, "kuro_siwo": 30},
        43: {"kuro_siwo": 10, "etci": 20, "mmflood": 30},
        44: {"mmflood": 10, "kuro_siwo": 20, "etci": 30},
    }
    if seed in rotations:
        return rotations[seed]
    # Default fall-through for custom seeds
    return {"etci": 10, "mmflood": 20, "kuro_siwo": 30}


def main():
    parser = argparse.ArgumentParser(description="Run cross-dataset federated learning experiment")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "fednova", "scaffold"],
        help="Federated optimization algorithm",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="main",
        choices=["main", "unequal"],
        help="Experiment setting (equal steps or unequal computation)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (42, 43, 44)")
    parser.add_argument("--manifest", type=str, default="data/mock/manifest.json", help="Path to manifest.json")
    parser.add_argument("--stats", type=str, default="data/mock/norm_stats.json", help="Path to norm_stats.json")
    parser.add_argument("--max_rounds", type=int, default=100, help="Maximum communication rounds")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in rounds")
    parser.add_argument("--fedprox_mu", type=float, default=0.01, help="FedProx proximal coefficient mu")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--local_steps", type=int, default=None, help="Override steps per round for testing")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/federated", help="Directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # Load client DataLoaders
    client_loaders = load_client_dataloaders(
        manifest_path=args.manifest,
        stats_path=args.stats,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Initialize client objects
    clients = []
    for client_id in ["etci", "mmflood", "kuro_siwo"]:
        loaders = client_loaders[client_id]
        train_ds = loaders["train"].dataset
        num_tiles = len(train_ds)
        client = FederatedClient(
            client_id=client_id,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
            num_train_tiles=num_tiles,
            device=device,
            learning_rate=0.01,
            weight_decay=1e-4,
            grad_norm_limit=5.0,
        )
        clients.append(client)

    # Initialize server
    out_dir = Path(args.checkpoint_dir) / f"{args.experiment}_{args.algorithm}"
    server = FederatedServer(
        clients=clients,
        device=device,
        seed=args.seed,
        algorithm=args.algorithm,
        fedprox_mu=args.fedprox_mu,
        max_rounds=args.max_rounds,
        early_stopping_patience=args.patience,
        checkpoint_dir=out_dir,
    )

    step_budgets = get_step_budgets(args.experiment, args.seed)
    if args.local_steps is not None:
        step_budgets = {k: args.local_steps for k in step_budgets}
    print(f"Step budgets per round: {step_budgets}", flush=True)

    results = server.run_training(step_budget_per_client=step_budgets)

    print("\n--- Test Evaluation Results ---")
    macro = results["test_results"]["macro"]
    pooled = results["test_results"]["pooled"]
    print(f"Macro Flood IoU: {macro['iou']:.4f} | Dice: {macro['dice']:.4f} | Prec: {macro['precision']:.4f} | Rec: {macro['recall']:.4f}")
    print(f"Pooled Flood IoU: {pooled['iou']:.4f} | Dice: {pooled['dice']:.4f}")
    for cid, m in results["test_results"]["per_client"].items():
        print(f"Client {cid:10s} -> IoU: {m['iou']:.4f}, Dice: {m['dice']:.4f}, Prec: {m['precision']:.4f}, Rec: {m['recall']:.4f}")


if __name__ == "__main__":
    main()
