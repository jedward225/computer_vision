from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.datasets import KitsSliceDataset
from src.models.factory import build_model
from src.utils.config import load_config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_experiment_and_data(exp_config: str | Path, data_config: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    exp_cfg = load_config(exp_config)
    data_cfg = load_config(data_config)
    return exp_cfg, data_cfg


def split_csv(data_cfg: dict[str, Any], split: str) -> Path:
    return Path(data_cfg["processed_root"]) / "splits" / f"{split}.csv"


def num_classes_for(binary: bool) -> int:
    return 2 if binary else 4


def make_loaders(
    exp_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    binary: bool = False,
    input_mode: str = "2d",
) -> tuple[DataLoader, DataLoader]:
    train_ds = KitsSliceDataset(split_csv(data_cfg, "train"), num_classes=4, input_mode=input_mode, augment=True, binary=binary)
    val_ds = KitsSliceDataset(split_csv(data_cfg, "val"), num_classes=4, input_mode=input_mode, augment=False, binary=binary)
    batch_size = int(exp_cfg["training"]["batch_size"])
    workers = int(exp_cfg["training"].get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def make_test_loader(
    exp_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    binary: bool = False,
    input_mode: str = "2d",
) -> DataLoader:
    test_ds = KitsSliceDataset(split_csv(data_cfg, "test"), num_classes=4, input_mode=input_mode, augment=False, binary=binary)
    batch_size = int(exp_cfg["training"].get("eval_batch_size", exp_cfg["training"]["batch_size"]))
    workers = int(exp_cfg["training"].get("num_workers", 4))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())


def build_experiment_model(exp_cfg: dict[str, Any], num_classes: int = 4) -> torch.nn.Module:
    cfg = json.loads(json.dumps(exp_cfg))
    name = cfg["model"]["name"]
    if name in {"unet_2d", "segresnet_2d"}:
        cfg["model"]["out_channels"] = num_classes
    elif name == "conditional_vae_seg":
        cfg["model"]["out_channels"] = num_classes
    elif name == "conditional_diffusion_unet":
        cfg["model"]["mask_channels"] = num_classes
        cfg["model"]["out_channels"] = num_classes
    return build_model(cfg)


def checkpoint_path(exp_cfg: dict[str, Any]) -> Path:
    return Path("checkpoints") / exp_cfg["experiment_name"] / "best.pt"


def log_path(exp_cfg: dict[str, Any], filename: str = "history.csv") -> Path:
    path = Path("logs") / exp_cfg["experiment_name"] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_history(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
