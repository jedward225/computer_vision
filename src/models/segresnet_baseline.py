from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.blocks import Downsample, ResBlock, Upsample, group_norm


class SegResNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: tuple[int, ...] | list[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        channels = list(channels)
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for idx, ch in enumerate(channels):
            in_ch = channels[idx - 1] if idx else channels[0]
            self.encoders.append(nn.Sequential(ResBlock(in_ch, ch), ResBlock(ch, ch)))
            if idx != len(channels) - 1:
                self.downs.append(Downsample(ch))

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = channels[-1]
        for ch in reversed(channels[:-1]):
            self.ups.append(Upsample(prev))
            self.decoders.append(nn.Sequential(ResBlock(prev + ch, ch), ResBlock(ch, ch)))
            prev = ch
        self.head = nn.Sequential(group_norm(prev), nn.SiLU(inplace=True), nn.Conv2d(prev, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        skips = []
        for idx, encoder in enumerate(self.encoders):
            h = encoder(h)
            if idx != len(self.encoders) - 1:
                skips.append(h)
                h = self.downs[idx](h)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)
        return self.head(h)

