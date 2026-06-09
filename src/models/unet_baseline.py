from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.blocks import ConvBlock


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: tuple[int, ...] | list[int] = (32, 64, 128, 256, 512),
    ) -> None:
        super().__init__()
        channels = list(channels)
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch

        self.pool = nn.MaxPool2d(2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(channels[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch))
            prev = ch
        self.head = nn.Conv2d(prev, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for idx, encoder in enumerate(self.encoders):
            h = encoder(h)
            if idx != len(self.encoders) - 1:
                skips.append(h)
                h = self.pool(h)

        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips), strict=True):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)
        return self.head(h)

