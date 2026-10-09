import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.models.unet import UNet, build_unet


def test_unet_shapes():
    model = UNet(in_channels=2, out_channels=1)
    x = torch.randn(2, 2, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"

    probs = model.predict_proba(x)
    assert probs.shape == (2, 1, 256, 256)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_unet_seed_initialization():
    m1 = build_unet(seed=42)
    m2 = build_unet(seed=42)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "Models with identical seed must match identically"


if __name__ == "__main__":
    test_unet_shapes()
    test_unet_seed_initialization()
    print("test_unet passed!")
