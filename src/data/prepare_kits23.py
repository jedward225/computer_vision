from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.utils.config import load_config


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KiTS23 2D/2.5D slices.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Optional explicit case IDs.")
    return parser.parse_args()


def _import_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is required for KiTS23 preprocessing. Install the project environment first."
        ) from exc
    return nib


def _resize_2d(image: np.ndarray, size: int, order: int) -> np.ndarray:
    if image.shape == (size, size):
        return image
    try:
        import cv2

        interpolation = cv2.INTER_LINEAR if order == 1 else cv2.INTER_NEAREST
        return cv2.resize(image, (size, size), interpolation=interpolation)
    except ImportError:
        from scipy.ndimage import zoom

        zoom_factors = (size / image.shape[0], size / image.shape[1])
        return zoom(image, zoom_factors, order=order)


def _normalize_case_id(path: Path) -> str:
    name = path.name
    if name.startswith("case_"):
        return name
    if name.isdigit():
        return f"case_{int(name):05d}"
    return name


def find_cases(data_root: Path) -> list[CaseFiles]:
    candidates: list[CaseFiles] = []
    for case_dir in sorted(data_root.rglob("*")):
        if not case_dir.is_dir():
            continue
        image_path = case_dir / "imaging.nii.gz"
        mask_path = case_dir / "segmentation.nii.gz"
        if image_path.exists() and mask_path.exists():
            candidates.append(CaseFiles(_normalize_case_id(case_dir), image_path, mask_path))
    return candidates


def hu_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    image = np.clip(image, low, high)
    image = (image - low) / max(high - low, 1e-6)
    return image.astype(np.float32)


def _slice_axis_from_orientation(nib, image_obj, requested_axis: str | int) -> int:
    if isinstance(requested_axis, int):
        if requested_axis < 0 or requested_axis >= len(image_obj.shape):
            raise ValueError(f"slice_axis {requested_axis} is out of bounds for shape {image_obj.shape}")
        return requested_axis

    axis_name = str(requested_axis).lower()
    if axis_name.isdigit():
        return _slice_axis_from_orientation(nib, image_obj, int(axis_name))

    targets = {
        "axial": {"i", "s"},
        "coronal": {"a", "p"},
        "sagittal": {"l", "r"},
    }
    if axis_name not in targets:
        raise ValueError(f"Unknown slice_axis {requested_axis!r}; expected axial, coronal, sagittal, or an integer axis")

    axcodes = tuple(str(code).lower() for code in nib.aff2axcodes(image_obj.affine))
    for axis, code in enumerate(axcodes):
        if code in targets[axis_name]:
            return axis
    raise ValueError(f"Could not infer {axis_name} axis from orientation codes {axcodes}")


def foreground_indices(mask: np.ndarray, slice_axis: int) -> np.ndarray:
    spatial_axes = tuple(axis for axis in range(mask.ndim) if axis != slice_axis)
    return np.where(np.any(mask > 0, axis=spatial_axes))[0]


def choose_slices(mask: np.ndarray, context_margin: int, empty_stride: int, slice_axis: int) -> list[int]:
    depth = mask.shape[slice_axis]
    selected: set[int] = set()
    fg = foreground_indices(mask, slice_axis=slice_axis)
    for z in fg.tolist():
        start = max(0, z - context_margin)
        end = min(depth - 1, z + context_margin)
        selected.update(range(start, end + 1))
    selected.update(range(0, depth, max(empty_stride, 1)))
    if not selected:
        selected.update(range(depth))
    return sorted(selected)


def take_slice(volume: np.ndarray, slice_axis: int, index: int) -> np.ndarray:
    return np.take(volume, indices=index, axis=slice_axis)


def split_cases(case_ids: list[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio))) if n else 0
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else max(0, n - n_train)
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "slice_index", "path", "has_foreground"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def process_case(
    case: CaseFiles,
    output_dir: Path,
    image_size: int,
    window_cfg: dict[str, float],
    keep_cfg: dict[str, int],
    slice_axis_cfg: str | int,
) -> list[dict[str, object]]:
    nib = _import_nibabel()
    image_obj = nib.load(str(case.image_path))
    mask_obj = nib.load(str(case.mask_path))
    image = np.asarray(image_obj.get_fdata(dtype=np.float32))
    mask = np.asarray(mask_obj.get_fdata(dtype=np.float32)).astype(np.uint8)

    if image.shape != mask.shape:
        raise ValueError(f"Shape mismatch for {case.case_id}: image {image.shape}, mask {mask.shape}")
    if image.ndim != 3:
        raise ValueError(f"Expected 3D volume for {case.case_id}, got shape {image.shape}")

    slice_axis = _slice_axis_from_orientation(nib, image_obj, slice_axis_cfg)
    image = hu_window(image, center=window_cfg["center"], width=window_cfg["width"])
    slices = choose_slices(
        mask,
        context_margin=int(keep_cfg.get("context_margin", 3)),
        empty_stride=int(keep_cfg.get("empty_stride", 10)),
        slice_axis=slice_axis,
    )

    case_out = output_dir / "slices" / case.case_id
    case_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for z in slices:
        image_z = _resize_2d(take_slice(image, slice_axis, z), image_size, order=1).astype(np.float16)
        mask_z = _resize_2d(take_slice(mask, slice_axis, z), image_size, order=0).astype(np.uint8)
        out_path = case_out / f"{case.case_id}_z{z:04d}.npz"
        np.savez(out_path, image=image_z, mask=mask_z, case_id=case.case_id, slice_index=z)
        rows.append(
            {
                "case_id": case.case_id,
                "slice_index": z,
                "path": str(out_path),
                "has_foreground": int(np.any(mask_z > 0)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"])
    processed_root = Path(cfg["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)

    cases = find_cases(data_root)
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [case for case in cases if case.case_id in wanted or case.case_id.replace("case_", "") in wanted]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise FileNotFoundError(
            f"No KiTS23 cases found under {data_root}. Expected case_xxxxx/imaging.nii.gz and segmentation.nii.gz."
        )

    print(f"Found {len(cases)} KiTS23 cases under {data_root}")
    all_rows_by_case: dict[str, list[dict[str, object]]] = {}
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] processing {case.case_id}")
        rows = process_case(
            case,
            processed_root,
            int(cfg["image_size"]),
            cfg["preprocess"]["hu_window"],
            cfg["preprocess"]["keep_slices"],
            cfg["preprocess"].get("slice_axis", cfg["preprocess"].get("axis", "axial")),
        )
        all_rows_by_case[case.case_id] = rows

    split_cfg = cfg["split"]
    splits = split_cases(
        list(all_rows_by_case.keys()),
        seed=int(cfg.get("seed", 2026)),
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
    )

    counts: dict[str, dict[str, int]] = {}
    for split_name, case_ids in splits.items():
        rows = [row for case_id in case_ids for row in all_rows_by_case[case_id]]
        n_rows = write_csv(processed_root / "splits" / f"{split_name}.csv", rows)
        counts[split_name] = {
            "cases": len(case_ids),
            "slices": n_rows,
            "foreground_slices": int(sum(int(row["has_foreground"]) for row in rows)),
        }

    metadata = {
        "data_root": str(data_root),
        "processed_root": str(processed_root),
        "image_size": int(cfg["image_size"]),
        "num_cases": len(cases),
        "splits": counts,
        "classes": cfg["classes"],
        "slice_axis": cfg["preprocess"].get("slice_axis", cfg["preprocess"].get("axis", "axial")),
    }
    with (processed_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
