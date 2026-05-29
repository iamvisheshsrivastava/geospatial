from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SatelliteAutoencoder(nn.Module):
    """Convolutional autoencoder for unsupervised anomaly detection on satellite patches.

    Input:  (B, 3, H, W) — ImageNet-normalised RGB
    Output: (B, 3, H, W) — reconstructed image in the same space

    Anomaly score = mean squared reconstruction error per sample.
    High score → patch looks unlike the training distribution → anomalous.
    """

    def __init__(self, latent_dim: int = 512) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 3×224×224 → latent_dim×1×1 (via adaptive pool)
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),            # 224
            ConvBlock(32, 64, stride=2), # 112
            ConvBlock(64, 128, stride=2),# 56
            ConvBlock(128, 256, stride=2),# 28
            ConvBlock(256, latent_dim, stride=2), # 14
            nn.AdaptiveAvgPool2d(1),     # 1×1
        )

        # Decoder: latent_dim×1×1 → 3×224×224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0),  # 7
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            ConvTransposeBlock(256, 128),  # 14
            ConvTransposeBlock(128, 64),   # 28
            ConvTransposeBlock(64, 32),    # 56
            ConvTransposeBlock(32, 16),    # 112
            ConvTransposeBlock(16, 8),     # 224
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Tanh(),  # output in [-1, 1] — compatible with ImageNet normalisation range
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # (B, latent_dim, 1, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # (B, 3, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction. Shape: (B,)"""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=(1, 2, 3))

    def anomaly_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel squared error averaged across channels. Shape: (B, H, W)"""
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=1)  # (B, H, W)


def build_autoencoder(latent_dim: int = 512) -> SatelliteAutoencoder:
    return SatelliteAutoencoder(latent_dim=latent_dim)
