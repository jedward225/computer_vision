from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.diffusion.scheduler import DiffusionScheduler
from src.engine.common import build_experiment_model, checkpoint_path, get_device, load_experiment_and_data, make_test_loader
from src.utils.checkpoint import load_checkpoint
from src.utils.visualization import mask_to_rgb, overlay_mask, save_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative segmentation figures.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--data-config", default="src/configs/dataset.yaml", help="Dataset YAML config.")
    parser.add_argument("--output-dir", default="results/figures/qualitative", help="Output directory.")
    parser.add_argument("--num-images", type=int, default=10, help="Number of test images to visualize.")
    parser.add_argument("--sample-steps", type=int, default=25, help="DDIM steps for diffusion models.")
    parser.add_argument("--uncertainty-samples", type=int, default=8, help="Repeated samples for uncertainty maps.")
    parser.add_argument("--denoising", action="store_true", help="Save denoising trajectory for diffusion models.")
    parser.add_argument("--uncertainty", action="store_true", help="Save uncertainty maps for diffusion models.")
    parser.add_argument("--foreground-only", action="store_true", help="Only save slices with foreground labels.")
    parser.add_argument("--min-foreground-pixels", type=int, default=0, help="Skip slices with fewer foreground pixels in the ground-truth mask.")
    parser.add_argument("--max-per-case", type=int, default=0, help="Maximum saved slices per case; 0 disables the limit.")
    parser.add_argument("--binary", action="store_true", help="Use binary foreground model.")
    return parser.parse_args()


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

    scheduler = None
    if exp_cfg["model"]["name"] == "conditional_diffusion_unet":
        scheduler = DiffusionScheduler(
            timesteps=int(exp_cfg["diffusion"]["train_steps"]),
            beta_schedule=str(exp_cfg["diffusion"].get("beta_schedule", "cosine")),
        ).to(device)
    prediction_type = str(exp_cfg.get("diffusion", {}).get("prediction_type", "epsilon"))

    output_dir = Path(args.output_dir) / exp_cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    saved_by_case: dict[str, int] = {}
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        if scheduler is not None:
            if args.denoising:
                logits, trajectory = scheduler.ddim_sample(
                    model,
                    image,
                    mask_channels=num_classes,
                    sample_steps=args.sample_steps,
                    return_trajectory=True,
                    prediction_type=prediction_type,
                )
            else:
                logits = scheduler.ddim_sample(
                    model,
                    image,
                    mask_channels=num_classes,
                    sample_steps=args.sample_steps,
                    prediction_type=prediction_type,
                )
                trajectory = []
        elif exp_cfg["model"]["name"] == "conditional_vae_seg":
            logits = model(image)["logits"]
            trajectory = []
        else:
            logits = model(image)
            trajectory = []

        pred = logits.argmax(dim=1)
        for i in range(image.shape[0]):
            if saved >= args.num_images:
                break
            if args.foreground_only and int(batch["has_foreground"][i]) == 0:
                continue
            case_id = batch["case_id"][i]
            if args.max_per_case and saved_by_case.get(case_id, 0) >= args.max_per_case:
                continue
            image_np = image[i].detach().cpu().numpy()
            mask_np = mask[i].detach().cpu().numpy()
            pred_np = pred[i].detach().cpu().numpy()
            if args.min_foreground_pixels and int((mask_np > 0).sum()) < args.min_foreground_pixels:
                continue
            slice_idx = int(batch["slice_index"][i])
            name = f"{saved:03d}_{case_id}_z{slice_idx:04d}"
            save_panel(
                output_dir / f"{name}.png",
                [
                    image_np[0],
                    mask_to_rgb(mask_np),
                    mask_to_rgb(pred_np),
                    overlay_mask(image_np, pred_np),
                ],
                ["Image", "Ground Truth", "Prediction", "Overlay"],
            )

            if trajectory:
                traj_images = []
                titles = []
                stride = max(1, len(trajectory) // 5)
                for step_idx in list(range(0, len(trajectory), stride))[:5]:
                    traj_pred = trajectory[step_idx][i].argmax(dim=0).numpy()
                    traj_images.append(mask_to_rgb(traj_pred))
                    titles.append(f"step {step_idx + 1}")
                save_panel(output_dir / f"{name}_denoising.png", traj_images, titles)

            if args.uncertainty and scheduler is not None:
                probs = []
                single_image = image[i : i + 1]
                for _ in range(args.uncertainty_samples):
                    sample_logits = scheduler.ddim_sample(
                        model,
                        single_image,
                        mask_channels=num_classes,
                        sample_steps=args.sample_steps,
                        prediction_type=prediction_type,
                    )
                    probs.append(sample_logits.softmax(dim=1).detach().cpu())
                prob = torch.cat(probs, dim=0).mean(dim=0)
                entropy = -(prob * torch.clamp(prob, min=1e-6).log()).sum(dim=0).numpy()
                entropy = entropy / max(float(entropy.max()), 1e-6)
                save_panel(
                    output_dir / f"{name}_uncertainty.png",
                    [image_np[0], entropy],
                    ["Image", "Entropy"],
                )

            saved += 1
            saved_by_case[case_id] = saved_by_case.get(case_id, 0) + 1
        if saved >= args.num_images:
            break
    print(f"saved {saved} qualitative figures to {output_dir}")


if __name__ == "__main__":
    main()
