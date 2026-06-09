from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 1.0, ce_weight: float = 1.0, include_background: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        probs = logits.softmax(dim=1)
        one_hot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        if not self.include_background and self.num_classes > 1:
            dice = dice[1:]
        dice_loss = 1 - dice.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

