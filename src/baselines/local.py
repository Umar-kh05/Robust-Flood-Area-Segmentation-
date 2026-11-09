from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.unet import build_unet
from ..losses.flood_loss import FloodLoss
from ..metrics.flood_metrics import ConfusionMatrixAccumulator
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


def train_local_model(
    client_id: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    seed: int = 42,
    max_steps: int = 2000,
    eval_interval: int = 20,
    patience: int = 15,
    checkpoint_dir: str | Path = "checkpoints/local",
) -> dict:
    """
    Trains a local-only reference model for a single client.
    Evaluates every 20 steps, patience 15, saves best client validation IoU.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"local_{client_id}_seed{seed}_best.pt"

    model = build_unet(seed=seed).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
    criterion = FloodLoss()
    sampler = StepBasedBatchSampler(train_loader)

    best_val_iou = -1.0
    best_step = -1
    no_improve_count = 0
    history = []

    print(f"Starting local training for {client_id} (Seed: {seed}, Max steps: {max_steps})")

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
            val_metrics = evaluate_dataloader(model, val_loader, device)
            val_iou = val_metrics["iou"]
            history.append({"step": step, "val_iou": val_iou, "val_metrics": val_metrics})

            print(f"Step {step:04d}/{max_steps} | Val IoU: {val_iou:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_step = step
                no_improve_count = 0
                torch.save({"step": step, "state_dict": model.state_dict(), "val_iou": val_iou}, best_path)
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at step {step} for {client_id}")
                    break

    # Load best checkpoint and evaluate on untouched test set
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_dataloader(model, test_loader, device)

    result = {
        "client_id": client_id,
        "seed": seed,
        "best_step": best_step,
        "best_val_iou": best_val_iou,
        "test_metrics": test_metrics,
        "history": history,
    }

    result_path = checkpoint_dir / f"local_{client_id}_seed{seed}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
