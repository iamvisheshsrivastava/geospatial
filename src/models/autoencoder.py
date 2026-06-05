from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _BaseAutoencoder(nn.Module):
    """Shared interface for all CAE variants."""

    encoder: nn.Sequential
    decoder: nn.Sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean BCE between input and reconstruction. Shape: (B,)"""
        with torch.no_grad():
            recon = self.forward(x)
            bce = nn.functional.binary_cross_entropy(recon, x, reduction="none")
            return bce.mean(dim=(1, 2, 3))

    def anomaly_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel BCE averaged across channels. Shape: (B, H, W)"""
        with torch.no_grad():
            recon = self.forward(x)
            bce = nn.functional.binary_cross_entropy(recon, x, reduction="none")
            return bce.mean(dim=1)


class CAE2Conv(_BaseAutoencoder):
    """2-conv-block encoder/decoder (shallowest variant from the paper)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(3, 64, stride=2),   # 112
            _conv_block(64, 128, stride=2),  # 56
            nn.AdaptiveAvgPool2d(1),         # 1×1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=1, padding=0),  # 7
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            _up_block(64, 32),   # 14
            _up_block(32, 16),   # 28
            _up_block(16, 8),    # 56
            _up_block(8, 4),     # 112
            _up_block(4, 3),     # 224
            nn.Sigmoid(),
        )


class CAE3Conv(_BaseAutoencoder):
    """3-conv-block encoder/decoder (medium depth, paper default)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(3, 32, stride=2),    # 112
            _conv_block(32, 64, stride=2),   # 56
            _conv_block(64, 128, stride=2),  # 28
            nn.AdaptiveAvgPool2d(1),         # 1×1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=7, stride=1, padding=0),  # 7
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            _up_block(128, 64),  # 14
            _up_block(64, 32),   # 28
            _up_block(32, 16),   # 56
            _up_block(16, 8),    # 112
            _up_block(8, 3),     # 224
            nn.Sigmoid(),
        )


class CAEVariedFilter(_BaseAutoencoder):
    """Varied-filter encoder (large→small filter progression, paper variant 3)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),   # 112
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 56
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False), # 28
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 1×1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=7, stride=1, padding=0),  # 7
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            _up_block(128, 64),  # 14
            _up_block(64, 32),   # 28
            _up_block(32, 16),   # 56
            _up_block(16, 8),    # 112
            _up_block(8, 3),     # 224
            nn.Sigmoid(),
        )


# Legacy alias kept so existing load_autoencoder calls don't break
SatelliteAutoencoder = CAE3Conv

ARCHITECTURES: dict[str, type[_BaseAutoencoder]] = {
    "CAE-2Conv": CAE2Conv,
    "CAE-3Conv": CAE3Conv,
    "CAE-VariedFilter": CAEVariedFilter,
}


def build_autoencoder(arch: str = "CAE-3Conv", **_kwargs) -> _BaseAutoencoder:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch]()
