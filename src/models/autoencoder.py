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


def _decoder_params(image_size: int) -> tuple[int, int]:
    """Return (init_size, n_up_blocks) so init_size * 2**n_up_blocks == image_size.

    Works by repeatedly halving image_size while it is even, up to 5 times.
    Examples: 224 → (7, 5)  because 7*32=224
              64  → (2, 5)  because 2*32=64
              128 → (4, 5)  because 4*32=128
    """
    n, s = 0, image_size
    while s % 2 == 0 and n < 5:
        s //= 2
        n += 1
    return s, n


def _build_decoder(
    bottleneck_ch: int,
    image_size: int,
    channel_seq: list[int],
) -> nn.Sequential:
    """Build a decoder that produces (3, image_size, image_size) from (bottleneck_ch, 1, 1)."""
    init_size, n_up_blocks = _decoder_params(image_size)

    layers: list[nn.Module] = [
        nn.ConvTranspose2d(bottleneck_ch, channel_seq[0],
                           kernel_size=init_size, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(channel_seq[0]),
        nn.ReLU(inplace=True),
    ]
    for i in range(n_up_blocks):
        out_ch = channel_seq[i + 1] if i + 1 < len(channel_seq) else 3
        layers.append(_up_block(channel_seq[i], out_ch))

    # Final 1×1 conv to exactly 3 channels if not already there
    if channel_seq[-1] != 3:
        layers.append(nn.Conv2d(channel_seq[-1], 3, kernel_size=1, bias=False))

    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


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
    """2-conv-block encoder — shallowest variant from the paper."""

    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(3, 64, stride=2),
            _conv_block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.decoder = _build_decoder(
            bottleneck_ch=128,
            image_size=image_size,
            channel_seq=[64, 32, 16, 8, 4, 3],
        )


class CAE3Conv(_BaseAutoencoder):
    """3-conv-block encoder — medium depth, paper default."""

    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(3, 32, stride=2),
            _conv_block(32, 64, stride=2),
            _conv_block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.decoder = _build_decoder(
            bottleneck_ch=128,
            image_size=image_size,
            channel_seq=[128, 64, 32, 16, 8, 3],
        )


class CAEVariedFilter(_BaseAutoencoder):
    """Varied-filter encoder — 5×5→3×3→1×1 kernel progression, paper variant 3."""

    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.decoder = _build_decoder(
            bottleneck_ch=128,
            image_size=image_size,
            channel_seq=[128, 64, 32, 16, 8, 3],
        )


# Legacy alias
SatelliteAutoencoder = CAE3Conv

ARCHITECTURES: dict[str, type[_BaseAutoencoder]] = {
    "CAE-2Conv": CAE2Conv,
    "CAE-3Conv": CAE3Conv,
    "CAE-VariedFilter": CAEVariedFilter,
}


def build_autoencoder(arch: str = "CAE-3Conv", image_size: int = 224) -> _BaseAutoencoder:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch](image_size=image_size)
