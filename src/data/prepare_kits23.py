from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess KiTS23 volumes into 2D axial slices.")
    parser.add_argument("--config", default="src/configs/dataset.yaml", help="Dataset YAML config.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional case limit for quick experiments.")
    return parser.parse_args()


def window_ct(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    return np.clip((volume - low) / (high - low), 0.0, 1.0).astype(np.float32)


def resize_image(image: np.ndarray, size: int, order: int) -> np.ndarray:
    return resize(
        image,
        (size, size),
        order=order,
        preserve_range=True,
        anti_aliasing=order != 0,
    ).astype(np.float32 if order != 0 else np.int64)


def case_dirs(data_root: Path) -> list[Path]:
    cases = sorted(p for p in data_root.glob("case_*") if (p / "imaging.nii.gz").exists() and (p / "segmentation.nii.gz").exists())
    if not cases:
        raise FileNotFoundError(f"No KiTS23 cases found under {data_root}")
    return cases


def split_cases(cases: list[Path], cfg: dict[str, Any]) -> dict[str, list[Path]]:
    rng = random.Random(int(cfg.get("seed", 2026)))
    cases = list(cases)
    rng.shuffle(cases)
    split_cfg = cfg["split"]
    n = len(cases)
    n_train = max(1, int(n * float(split_cfg["train_ratio"])))
    n_val = max(1, int(n * float(split_cfg["val_ratio"]))) if n >= 3 else 0
    return {
        "train": cases[:n_train],
        "val": cases[n_train : n_train + n_val],
        "test": cases[n_train + n_val :],
    }


def selected_slices(mask: np.ndarray, cfg: dict[str, Any]) -> set[int]:
    keep_cfg = cfg["preprocess"]["keep_slices"]
    margin = int(keep_cfg.get("context_margin", 3))
    stride = int(keep_cfg.get("empty_stride", 10))
    foreground = np.where(mask.reshape(-1, mask.shape[-1]).max(axis=0) > 0)[0]
    keep: set[int] = set()
    depth = mask.shape[-1]
    for z in foreground:
        for zz in range(max(0, int(z) - margin), min(depth, int(z) + margin + 1)):
            keep.add(zz)
    if keep_cfg.get("mode", "foreground_or_context") == "foreground_or_context":
        keep.update(range(0, depth, stride))
    return keep


def preprocess_case(case_dir: Path, split: str, cfg: dict[str, Any], output_root: Path) -> list[dict[str, str | int]]:
    image = nib.load(str(case_dir / "imaging.nii.gz")).get_fdata(dtype=np.float32)
    mask = nib.load(str(case_dir / "segmentation.nii.gz")).get_fdata(dtype=np.float32).astype(np.int64)
    hu = cfg["preprocess"]["hu_window"]
    image = window_ct(image, float(hu["center"]), float(hu["width"]))
    size = int(cfg["image_size"])
    keep = selected_slices(mask, cfg)
    case_id = case_dir.name
    image_dir = output_root / "images" / case_id
    mask_dir = output_root / "masks" / case_id
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    depth = image.shape[-1]

    for z in sorted(keep):
        img = resize_image(image[:, :, z], size, order=1)
        msk = resize_image(mask[:, :, z], size, order=0).clip(0, 3)
        image_path = image_dir / f"{z:04d}.npy"
        mask_path = mask_dir / f"{z:04d}.npy"
        np.save(image_path, img.astype(np.float32))
        np.save(mask_path, msk.astype(np.int64))
        prev_z = max(0, z - 1)
        next_z = min(depth - 1, z + 1)
        prev_path = image_dir / f"{prev_z:04d}.npy"
        next_path = image_dir / f"{next_z:04d}.npy"
        if not prev_path.exists():
            np.save(prev_path, resize_image(image[:, :, prev_z], size, order=1).astype(np.float32))
        if not next_path.exists():
            np.save(next_path, resize_image(image[:, :, next_z], size, order=1).astype(np.float32))
        rows.append(
            {
                "case_id": case_id,
                "slice_idx": z,
                "split": split,
                "image_path": str(image_path),
                "image_prev_path": str(prev_path),
                "image_next_path": str(next_path),
                "mask_path": str(mask_path),
            }
        )
    return rows


def write_split_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "slice_idx", "split", "image_path", "image_prev_path", "image_next_path", "mask_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"])
    output_root = Path(cfg["processed_root"])
    cases = case_dirs(data_root)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    splits = split_cases(cases, cfg)
    for split, split_cases_ in splits.items():
        rows: list[dict[str, str | int]] = []
        for case_dir in tqdm(split_cases_, desc=f"preprocess {split}"):
            rows.extend(preprocess_case(case_dir, split, cfg, output_root))
        write_split_csv(output_root / "splits" / f"{split}.csv", rows)
        print(f"{split}: {len(split_cases_)} cases, {len(rows)} slices")


if __name__ == "__main__":
    main()

