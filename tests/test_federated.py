import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tempfile
import torch
import torch.nn as nn

from src.data.preprocess import generate_mock_dataset
from src.data.dataset import load_client_dataloaders
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer


class TinyTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


def test_federated_algorithms_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        generate_mock_dataset(tmp_path, tiles_per_split=2)

        manifest_path = tmp_path / "manifest.json"
        stats_path = tmp_path / "norm_stats.json"

        client_loaders = load_client_dataloaders(
            manifest_path=manifest_path,
            stats_path=stats_path,
            batch_size=2,
            seed=42,
        )

        device = torch.device("cpu")

        for algo in ["fedavg", "fedprox", "fednova", "scaffold"]:
            clients = []
            for cid in ["etci", "mmflood", "kuro_siwo"]:
                loaders = client_loaders[cid]
                c = FederatedClient(
                    client_id=cid,
                    train_loader=loaders["train"],
                    val_loader=loaders["val"],
                    test_loader=loaders["test"],
                    num_train_tiles=len(loaders["train"].dataset),
                    device=device,
                    learning_rate=0.01,
                    weight_decay=1e-4,
                )
                clients.append(c)

            test_model = TinyTestNet()
            server = FederatedServer(
                clients=clients,
                device=device,
                seed=42,
                algorithm=algo,
                max_rounds=1,
                early_stopping_patience=1,
                checkpoint_dir=tmp_path / f"ckpt_{algo}",
                model=test_model,
            )

            # 1 local step per client for quick smoke test
            step_budgets = {"etci": 1, "mmflood": 1, "kuro_siwo": 1}
            summary = server.run_training(step_budgets)

            assert summary["best_round"] >= 1
            assert "macro" in summary["test_results"]
            print(f"Algorithm {algo} passed!")


if __name__ == "__main__":
    test_federated_algorithms_end_to_end()
    print("All federated algorithm tests passed!")
