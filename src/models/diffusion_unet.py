from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.blocks import Downsample, ResBlock, Upsample, group_norm, sinusoidal_embedding


class ConditionalDiffusionUNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        mask_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        time_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim or base_channels * 4
        channels = [base_channels * mult for mult in channel_mult]
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.stem = nn.Conv2d(image_channels + mask_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = channels[0]
        for idx, ch in enumerate(channels):
            self.down_blocks.append(nn.ModuleList([ResBlock(prev, ch, self.time_dim), ResBlock(ch, ch, self.time_dim)]))
            prev = ch
            if idx != len(channels) - 1:
                self.downsamples.append(Downsample(ch))

        self.mid1 = ResBlock(channels[-1], channels[-1], self.time_dim)
        self.mid2 = ResBlock(channels[-1], channels[-1], self.time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        prev = channels[-1]
        for ch in reversed(channels[:-1]):
            self.upsamples.append(Upsample(prev))
            self.up_blocks.append(nn.ModuleList([ResBlock(prev + ch, ch, self.time_dim), ResBlock(ch, ch, self.time_dim)]))
            prev = ch

        self.head = nn.Sequential(group_norm(prev), nn.SiLU(inplace=True), nn.Conv2d(prev, out_channels, 1))
        self.base_channels = base_channels

    def forward(self, image: torch.Tensor, noisy_mask: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(timesteps, self.base_channels)
        t_emb = self.time_mlp(t_emb)

        h = self.stem(torch.cat([image, noisy_mask], dim=1))
        skips = []
        for idx, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            if idx != len(self.down_blocks) - 1:
                skips.append(h)
                h = self.downsamples[idx](h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for up, blocks, skip in zip(self.upsamples, self.up_blocks, reversed(skips), strict=True):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
        return self.head(h)

