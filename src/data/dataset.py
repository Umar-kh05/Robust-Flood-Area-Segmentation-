from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import NormalizeSAR, RandomSARAugmentation, PadTo256


class FloodDataset(Dataset):
    """
    Dataset representing 256x256 2-channel (VV, VH) Sentinel-1 tiles.
    Loads pre-extracted or cached tile archives (.npz or .npy).
    """

    def __init__(
        self,
        samples: list[dict],
        normalization: NormalizeSAR | None = None,
        is_train: bool = False,
    ):
        self.samples = samples
        self.normalization = normalization
        self.is_train = is_train
        self.augment = RandomSARAugmentation() if is_train else None
        self.padder = PadTo256(target_size=256, pad_value_mask=255)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.samples[idx]
        file_path = Path(item["path"])

        # Load tile data
        if file_path.suffix == ".npz":
            data = np.load(file_path)
            image = data["image"]  # (2, H, W)
            mask = data["mask"]    # (H, W)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Ensure spatial padding if needed
        image, mask = self.padder(image, mask)

        # Apply client-specific normalization
        if self.normalization is not None:
            valid_mask = mask != 255
            image = self.normalization(image, valid_mask=valid_mask)

        # Apply spatial augmentations for training
        if self.augment is not None:
            image, mask = self.augment(image, mask)

        image_t = torch.from_numpy(image.astype(np.float32))
        mask_t = torch.from_numpy(mask.astype(np.int64))

        return {
            "image": image_t,
            "mask": mask_t,
            "tile_id": item.get("tile_id", str(idx)),
            "client": item.get("client", "unknown"),
            "event_id": item.get("event_id", "unknown"),
        }


def create_dataloader(
    dataset: FloodDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    """Creates a standard DataLoader for evaluation or training."""
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
    )


class StepBasedBatchSampler:
    """
    Infinite iterator for step-based client training.
    Shuffles indices upon epoch completion, ensuring smooth multi-round step execution.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next_batch(self) -> dict[str, torch.Tensor | str]:
        while True:
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)
                batch = next(self.iterator)

            # Check if batch contains at least one valid pixel
            masks = batch["mask"]
            if (masks != 255).any():
                return batch


def load_client_dataloaders(
    manifest_path: str | Path,
    stats_path: str | Path,
    batch_size: int = 8,
    seed: int = 42,
) -> dict[str, dict[str, DataLoader]]:
    """
    Parses manifest.json and norm_stats.json and builds train, val, and test DataLoaders
    for each client ("etci", "mmflood", "kuro_siwo").
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    clients = ["etci", "mmflood", "kuro_siwo"]
    client_loaders = {}

    for client in clients:
        client_stats = stats.get(client, {})
        vv_limits = client_stats.get("vv", (-25.0, 0.0))
        vh_limits = client_stats.get("vh", (-30.0, -5.0))
        norm = NormalizeSAR(vv_limits=vv_limits, vh_limits=vh_limits)

        client_loaders[client] = {}
        for split in ["train", "val", "test"]:
            tiles = [it for it in manifest if it["client"] == client and it["split"] == split]
            is_train = split == "train"
            ds = FloodDataset(tiles, normalization=norm, is_train=is_train)
            loader = create_dataloader(
                ds,
                batch_size=batch_size,
                shuffle=is_train,
                seed=seed if is_train else None,
            )
            client_loaders[client][split] = loader

    return client_loaders


def load_pooled_train_loader(
    manifest_path: str | Path,
    stats_path: str | Path,
    batch_size: int = 8,
    seed: int = 42,
) -> DataLoader:
    """
    Builds a pooled training DataLoader combining training portions across all three clients.
    Uses each client's source-specific normalization transform without pooling normalization.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    train_datasets = []
    clients = ["etci", "mmflood", "kuro_siwo"]

    for client in clients:
        client_stats = stats.get(client, {})
        vv_limits = client_stats.get("vv", (-25.0, 0.0))
        vh_limits = client_stats.get("vh", (-30.0, -5.0))
        norm = NormalizeSAR(vv_limits=vv_limits, vh_limits=vh_limits)

        tiles = [it for it in manifest if it["client"] == client and it["split"] == "train"]
        ds = FloodDataset(tiles, normalization=norm, is_train=True)
        train_datasets.append(ds)

    combined_ds = torch.utils.data.ConcatDataset(train_datasets)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        combined_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

