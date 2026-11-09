from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.unet import build_unet
from ..losses.flood_loss import FloodLoss
from ..metrics.flood_metrics import ConfusionMatrixAccumulator, compute_metrics_from_counts
from ..data.dataset import StepBasedBatchSampler


def evaluate_dataloader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    accum = ConfusionMatrixAccumulator(threshold=0.5, ignore_index=255)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            probs = model.predict_proba(images)
            accum.update(probs, masks)
    return accum.compute()


def train_centralized_model(
    pooled_train_loader: DataLoader,
    val_loaders: dict[str, DataLoader],
    test_loaders: dict[str, DataLoader],
    device: torch.device,
    seed: int = 42,
    max_steps: int = 6000,
    eval_interval: int = 60,
    patience: int = 15,
    checkpoint_dir: str | Path = "checkpoints/centralized",
) -> dict:
    """
    Trains a pooled centralized reference model.
    Evaluates every 60 steps, checkpoint selected by highest macro-mean client validation flood IoU.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"centralized_seed{seed}_best.pt"

    model = build_unet(seed=seed).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
    criterion = FloodLoss()
    sampler = StepBasedBatchSampler(pooled_train_loader)

    best_macro_val_iou = -1.0
    best_step = -1
    no_improve_count = 0
    history = []

    print(f"Starting centralized training (Seed: {seed}, Max steps: {max_steps})")

    for step in range(1, max_steps + 1):
        model.train()
        batch = sampler.next_batch()
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss, _ = criterion(logits, masks)
        if loss is None:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step % eval_interval == 0:
            client_ious = {}
            for cid, vloader in val_loaders.items():
                m = evaluate_dataloader(model, vloader, device)
                client_ious[cid] = m["iou"]

            macro_val_iou = sum(client_ious.values()) / len(client_ious)
            history.append({"step": step, "macro_val_iou": macro_val_iou, "client_ious": client_ious})

            print(
                f"Step {step:04d}/{max_steps} | Macro Val IoU: {macro_val_iou:.4f} | "
                + " | ".join(f"{c}: {iou:.4f}" for c, iou in client_ious.items())
            )

            if macro_val_iou > best_macro_val_iou:
                best_macro_val_iou = macro_val_iou
                best_step = step
                no_improve_count = 0
                torch.save(
                    {"step": step, "state_dict": model.state_dict(), "macro_val_iou": macro_val_iou},
                    best_path,
                )
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at step {step} for centralized model")
                    break

    # Load best checkpoint and evaluate on client test sets
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    per_client_test = {}
    pooled_tp, pooled_fp, pooled_fn, pooled_tn = 0, 0, 0, 0
    for cid, tloader in test_loaders.items():
        tm = evaluate_dataloader(model, tloader, device)
        per_client_test[cid] = tm
        pooled_tp += tm["tp"]
        pooled_fp += tm["fp"]
        pooled_fn += tm["fn"]
        pooled_tn += tm["tn"]

    macro_iou = sum(m["iou"] for m in per_client_test.values()) / len(per_client_test)
    macro_dice = sum(m["dice"] for m in per_client_test.values()) / len(per_client_test)
    macro_precision = sum(m["precision"] for m in per_client_test.values()) / len(per_client_test)
    macro_recall = sum(m["recall"] for m in per_client_test.values()) / len(per_client_test)
    pooled_test = compute_metrics_from_counts(pooled_tp, pooled_fp, pooled_fn, pooled_tn)

    result = {
        "seed": seed,
        "best_step": best_step,
        "best_macro_val_iou": best_macro_val_iou,
        "test_results": {
            "per_client": per_client_test,
            "macro": {
                "iou": macro_iou,
                "dice": macro_dice,
                "precision": macro_precision,
                "recall": macro_recall,
            },
            "pooled": pooled_test,
        },
        "history": history,
    }

    result_path = checkpoint_dir / f"centralized_seed{seed}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
