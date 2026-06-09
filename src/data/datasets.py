from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SliceRecord:
    case_id: str
    slice_index: int
    path: Path
    has_foreground: bool


def read_split_csv(path: str | Path) -> list[SliceRecord]:
    records: list[SliceRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                SliceRecord(
                    case_id=row["case_id"],
                    slice_index=int(row["slice_index"]),
                    path=Path(row["path"]),
                    has_foreground=bool(int(row["has_foreground"])),
                )
            )
    return records


def _neighbor_path(record: SliceRecord, z: int) -> Path:
    return record.path.parent / f"{record.case_id}_z{z:04d}.npz"


class KitsSliceDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        num_classes: int = 4,
        input_mode: str = "2d",
        augment: bool = False,
        binary: bool = False,
    ) -> None:
        self.records = read_split_csv(split_csv)
        self.num_classes = 2 if binary else num_classes
        self.input_mode = input_mode
        self.augment = augment
        self.binary = binary

    def __len__(self) -> int:
        return len(self.records)

    def _load_npz(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path) as data:
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.int64)
        if self.binary:
            mask = (mask > 0).astype(np.int64)
        return image, mask

    def _load_image_stack(self, record: SliceRecord) -> tuple[np.ndarray, np.ndarray]:
        image, mask = self._load_npz(record.path)
        if self.input_mode.lower() not in {"2.5d", "25d"}:
            return image[None, ...], mask

        images = []
        for dz in (-1, 0, 1):
            neighbor = _neighbor_path(record, record.slice_index + dz)
            if neighbor.exists():
                neighbor_image, _ = self._load_npz(neighbor)
                images.append(neighbor_image)
            else:
                images.append(image)
        return np.stack(images, axis=0), mask

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=-1)
            mask = np.flip(mask, axis=-1)
        if random.random() < 0.5:
            image = np.flip(image, axis=-2)
            mask = np.flip(mask, axis=-2)
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k=k, axes=(-2, -1))
            mask = np.rot90(mask, k=k, axes=(-2, -1))
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        record = self.records[index]
        image, mask = self._load_image_stack(record)
        if self.augment:
            image, mask = self._augment(image, mask)

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).long(),
            "case_id": record.case_id,
            "slice_index": record.slice_index,
            "has_foreground": int(record.has_foreground),
        }


def mask_to_one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(mask.long(), num_classes=num_classes)
    return one_hot.permute(0, 3, 1, 2).float()

