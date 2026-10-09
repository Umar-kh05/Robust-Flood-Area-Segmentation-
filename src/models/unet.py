import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8, eps: float = 1e-5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=eps),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=eps),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net with Group Normalization for 2-channel Sentinel-1 (VV, VH) flood segmentation.
    Follows configuration from Table 1:
      - 4 encoder stages: 32, 64, 128, 256
      - Bottleneck: 512 with 0.1 dropout
      - 4 decoder stages: 256, 128, 64, 32
      - 1x1 convolution output logit
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1, dropout: float = 0.1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(256, 512)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.dec4 = DecoderStage(512, 256, 256)
        self.dec3 = DecoderStage(256, 128, 128)
        self.dec2 = DecoderStage(128, 64, 64)
        self.dec1 = DecoderStage(64, 32, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))
        b = self.dropout(b)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


def build_unet(seed: int | None = None, in_channels: int = 2, out_channels: int = 1) -> UNet:
    """Builds a UNet model with reproducible weight initialization if seed is given."""
    if seed is not None:
        torch.manual_seed(seed)
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    return model


def get_model_weights_clone(model: nn.Module) -> dict[str, torch.Tensor]:
    """Returns deep copy of model state dict on CPU."""
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
