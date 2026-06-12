from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.diffusion.scheduler import DiffusionScheduler
from src.engine.common import build_experiment_model, checkpoint_path, get_device, load_experiment_and_data, make_test_loader
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import SegmentationMeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify uncertainty from repeated diffusion samples.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--data-config", default="src/configs/dataset.yaml", help="Dataset YAML config.")
    parser.add_argument("--output", default="results/tables/uncertainty_summary.csv", help="Output CSV path.")
    parser.add_argument("--sample-steps", type=int, default=5, help="DDIM steps per sample.")
    parser.add_argument("--uncertainty-samples", type=int, default=8, help="Repeated samples per image.")
    parser.add_argument("--max-images", type=int, default=128, help="Maximum test images to evaluate.")
    parser.add_argument("--foreground-only", action="store_true", help="Evaluate only slices with foreground labels.")
    parser.add_argument("--min-foreground-pixels", type=int, default=0, help="Skip slices with fewer foreground pixels.")
    parser.add_argument("--binary", action="store_true", help="Use binary foreground model.")
    return parser.parse_args()


def _select_indices(batch: dict, foreground_only: bool, min_foreground_pixels: int) -> list[int]:
    keep: list[int] = []
    mask = batch["mask"]
    for idx in range(mask.shape[0]):
        if foreground_only and int(batch["has_foreground"][idx]) == 0:
            continue
        if min_foreground_pixels and int((mask[idx] > 0).sum()) < min_foreground_pixels:
            continue
        keep.append(idx)
    return keep


@torch.no_grad()
def main() -> None:
    args = parse_args()
    exp_cfg, data_cfg = load_experiment_and_data(args.config, args.data_config)
    device = get_device()
    num_classes = 2 if args.binary else 4
    input_mode = exp_cfg.get("data", {}).get("input_mode", "2d")
    loader = make_test_loader(exp_cfg, data_cfg, binary=args.binary, input_mode=input_mode)

    model = build_experiment_model(exp_cfg, num_classes=num_classes).to(device)
    ckpt = load_checkpoint(checkpoint_path(exp_cfg), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if exp_cfg["model"]["name"] != "conditional_diffusion_unet":
        raise ValueError("Uncertainty summary expects a conditional_diffusion_unet config.")
    scheduler = DiffusionScheduler(
        timesteps=int(exp_cfg["diffusion"]["train_steps"]),
        beta_schedule=str(exp_cfg["diffusion"].get("beta_schedule", "cosine")),
    ).to(device)
    prediction_type = str(exp_cfg.get("diffusion", {}).get("prediction_type", "epsilon"))

    meter = SegmentationMeter(num_classes)
    total_images = 0
    total_pixels = 0
    entropy_sum = 0.0
    error_entropy_sum = 0.0
    correct_entropy_sum = 0.0
    foreground_entropy_sum = 0.0
    background_entropy_sum = 0.0
    error_count = 0
    correct_count = 0
    foreground_count = 0
    background_count = 0
    top_entropy_error_count = 0
    top_entropy_count = 0
    vote_disagreement_sum = 0.0

    progress = tqdm(loader, desc="uncertainty", leave=False)
    for batch in progress:
        keep = _select_indices(batch, args.foreground_only, args.min_foreground_pixels)
        if not keep:
            continue
        remaining = args.max_images - total_images
        if remaining <= 0:
            break
        keep = keep[:remaining]

        image = batch["image"][keep].to(device)
        mask = batch["mask"][keep].to(device)
        probs = []
        sample_preds = []
        for _ in range(args.uncertainty_samples):
            logits = scheduler.ddim_sample(
                model,
                image,
                mask_channels=num_classes,
                sample_steps=args.sample_steps,
                prediction_type=prediction_type,
            )
            prob = logits.softmax(dim=1)
            probs.append(prob)
            sample_preds.append(prob.argmax(dim=1))

        samples = torch.stack(probs, dim=0)
        mean_prob = samples.mean(dim=0)
        pred = mean_prob.argmax(dim=1)
        meter.update(pred, mask)

        entropy = -(mean_prob * mean_prob.clamp_min(1e-6).log()).sum(dim=1) / math.log(num_classes)
        error = pred != mask
        correct = ~error
        foreground = mask > 0
        background = ~foreground

        sample_pred = torch.stack(sample_preds, dim=0)
        vote_prob = F.one_hot(sample_pred, num_classes=num_classes).float().mean(dim=0)
        vote_disagreement = 1.0 - vote_prob.max(dim=-1).values

        total_images += image.shape[0]
        total_pixels += entropy.numel()
        entropy_sum += float(entropy.sum().item())
        error_entropy_sum += float(entropy[error].sum().item())
        correct_entropy_sum += float(entropy[correct].sum().item())
        foreground_entropy_sum += float(entropy[foreground].sum().item())
        background_entropy_sum += float(entropy[background].sum().item())
        error_count += int(error.sum().item())
        correct_count += int(correct.sum().item())
        foreground_count += int(foreground.sum().item())
        background_count += int(background.sum().item())
        vote_disagreement_sum += float(vote_disagreement.sum().item())

        flat_entropy = entropy.flatten()
        flat_error = error.flatten()
        top_k = max(1, int(0.10 * flat_entropy.numel()))
        top_idx = torch.topk(flat_entropy, top_k).indices
        top_entropy_error_count += int(flat_error[top_idx].sum().item())
        top_entropy_count += top_k
        progress.set_postfix(images=total_images)

    metrics = meter.compute()
    row = {
        "experiment": exp_cfg["experiment_name"],
        "sample_steps": args.sample_steps,
        "uncertainty_samples": args.uncertainty_samples,
        "num_images": total_images,
        "mean_dice": metrics["mean_dice"],
        "mean_entropy": entropy_sum / max(total_pixels, 1),
        "mean_vote_disagreement": vote_disagreement_sum / max(total_pixels, 1),
        "error_rate": error_count / max(total_pixels, 1),
        "top10_entropy_error_rate": top_entropy_error_count / max(top_entropy_count, 1),
        "error_entropy": error_entropy_sum / max(error_count, 1),
        "correct_entropy": correct_entropy_sum / max(correct_count, 1),
        "foreground_entropy": foreground_entropy_sum / max(foreground_count, 1),
        "background_entropy": background_entropy_sum / max(background_count, 1),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
