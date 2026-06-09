from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def mask_to_one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(mask.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


class KitsSliceDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        num_classes: int = 4,
        input_mode: str = "2d",
        augment: bool = False,
        binary: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.num_classes = num_classes
        self.input_mode = input_mode
        self.augment = augment
        self.binary = binary
        with self.split_csv.open("r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"Empty split file: {self.split_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: str) -> np.ndarray:
        return np.load(path).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[index]
        if self.input_mode == "2.5d":
            image = np.stack(
                [
                    self._load_image(row["image_prev_path"]),
                    self._load_image(row["image_path"]),
                    self._load_image(row["image_next_path"]),
                ],
                axis=0,
            )
        else:
            image = self._load_image(row["image_path"])[None, ...]

        mask = np.load(row["mask_path"]).astype(np.int64)
        if self.binary:
            mask = (mask > 0).astype(np.int64)

        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=-1).copy()
                mask = np.flip(mask, axis=-1).copy()
            if random.random() < 0.5:
                image = np.flip(image, axis=-2).copy()
                mask = np.flip(mask, axis=-2).copy()

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "case_id": row["case_id"],
            "slice_idx": int(row["slice_idx"]),
        }

