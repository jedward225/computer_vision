from __future__ import annotations

from typing import Any

from torch import nn

from src.models.cvae_seg import ConditionalVAESeg
from src.models.diffusion_unet import ConditionalDiffusionUNet
from src.models.segresnet_baseline import SegResNet2D
from src.models.unet_baseline import UNet2D


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    model_cfg.pop("spatial_dims", None)
    model_cfg.pop("strides", None)
    model_cfg.pop("num_res_units", None)
    model_cfg.pop("attention_resolutions", None)

    if name == "unet_2d":
        return UNet2D(**model_cfg)
    if name == "segresnet_2d":
        return SegResNet2D(**model_cfg)
    if name == "conditional_vae_seg":
        return ConditionalVAESeg(**model_cfg)
    if name == "conditional_diffusion_unet":
        return ConditionalDiffusionUNet(**model_cfg)
    raise ValueError(f"Unknown model name: {name}")
