from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.data.datasets import mask_to_one_hot
from src.models.blocks import ConvBlock


class ConditionalVAESeg(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        latent_dim: int = 64,
        encoder_channels: tuple[int, ...] | list[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        channels = list(encoder_channels)

        enc_layers: list[nn.Module] = []
        prev = in_channels + out_channels
        for ch in channels:
            enc_layers.extend([ConvBlock(prev, ch), nn.MaxPool2d(2)])
            prev = ch
        self.encoder = nn.Sequential(*enc_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mu = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

        self.image_encoder = ConvBlock(in_channels, channels[0])
        self.z_proj = nn.Linear(latent_dim, channels[0])
        self.decoder = nn.Sequential(
            ConvBlock(channels[0] * 2, channels[0]),
            ConvBlock(channels[0], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def encode(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot = mask_to_one_hot(mask, self.out_channels)
        h = self.encoder(torch.cat([image, one_hot], dim=1))
        h = self.pool(h).flatten(1)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, image: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        z_feat = self.z_proj(z)[:, :, None, None].expand(-1, -1, image.shape[-2], image.shape[-1])
        return self.decoder(torch.cat([image_feat, z_feat], dim=1))

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if mask is None:
            if self.training:
                z = torch.randn(image.shape[0], self.latent_dim, device=image.device)
            else:
                z = torch.zeros(image.shape[0], self.latent_dim, device=image.device)
            return {"logits": self.decode(image, z)}
        mu, logvar = self.encode(image, mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(image, z)
        return {"logits": logits, "mu": mu, "logvar": logvar}
