from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.diffusion.scheduler import DiffusionScheduler
from src.engine.common import build_experiment_model, checkpoint_path, get_device, load_experiment_and_data, make_test_loader
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.metrics import SegmentationMeter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained segmentation models.")
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment YAML configs.")
    parser.add_argument("--data-config", default="src/configs/dataset.yaml", help="Path to dataset YAML config.")
    parser.add_argument("--binary", action="store_true", help="Evaluate binary foreground segmentation.")
    parser.add_argument("--output", default="results/tables/evaluation.csv", help="Output CSV path.")
    return parser.parse_args()


@torch.no_grad()
def evaluate_config(config_path: str, data_config: str, binary: bool, device: torch.device) -> list[dict[str, float | str | int]]:
    exp_cfg, data_cfg = load_experiment_and_data(config_path, data_config)
    ckpt_path = checkpoint_path(exp_cfg)
    if not ckpt_path.exists():
        print(f"skip {exp_cfg['experiment_name']}: missing checkpoint {ckpt_path}")
        return []

    num_classes = 2 if binary else 4
    input_mode = exp_cfg.get("data", {}).get("input_mode", "2d")
    loader = make_test_loader(exp_cfg, data_cfg, binary=binary, input_mode=input_mode)
    model = build_experiment_model(exp_cfg, num_classes=num_classes).to(device)
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    model_name = exp_cfg["model"]["name"]
    rows: list[dict[str, float | str | int]] = []
    if model_name == "conditional_diffusion_unet":
        scheduler = DiffusionScheduler(
            timesteps=int(exp_cfg["diffusion"]["train_steps"]),
            beta_schedule=str(exp_cfg["diffusion"].get("beta_schedule", "cosine")),
        ).to(device)
        sample_steps_list = exp_cfg["diffusion"].get("sample_steps", [25])
    else:
        scheduler = None
        sample_steps_list = [1]

    for sample_steps in sample_steps_list:
        meter = SegmentationMeter(num_classes)
        total_time = 0.0
        total_images = 0
        for batch in tqdm(loader, desc=f"eval {exp_cfg['experiment_name']} steps={sample_steps}", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            start = time.perf_counter()
            if model_name == "conditional_diffusion_unet":
                assert scheduler is not None
                logits = scheduler.ddim_sample(model, image, mask_channels=num_classes, sample_steps=int(sample_steps))
            elif model_name == "conditional_vae_seg":
                logits = model(image)["logits"]
            else:
                logits = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start
            total_images += image.shape[0]
            pred = logits.argmax(dim=1)
            meter.update(pred, mask)

        metrics = meter.compute()
        rows.append(
            {
                "experiment": exp_cfg["experiment_name"],
                "model": model_name,
                "sample_steps": int(sample_steps),
                "num_classes": num_classes,
                "mean_dice": metrics["mean_dice"],
                "mean_iou": metrics["mean_iou"],
                "dice_kidney": metrics.get("dice_class_1", 0.0),
                "dice_tumor": metrics.get("dice_class_2", 0.0),
                "dice_cyst": metrics.get("dice_class_3", 0.0),
                "ms_per_image": 1000.0 * total_time / max(total_images, 1),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = get_device()
    rows: list[dict[str, float | str | int]] = []
    for config_path in args.configs:
        # Validate the config path early so typos are reported even when checkpoints are missing.
        load_config(config_path)
        rows.extend(evaluate_config(config_path, args.data_config, args.binary, device))

    if not rows:
        print("No evaluation rows produced.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
