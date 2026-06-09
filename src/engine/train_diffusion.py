from __future__ import annotations

import argparse

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from tqdm import tqdm

from src.data.datasets import mask_to_one_hot
from src.diffusion.scheduler import DiffusionScheduler
from src.engine.common import (
    append_history,
    build_experiment_model,
    checkpoint_path,
    get_device,
    load_experiment_and_data,
    log_path,
    make_loaders,
)
from src.engine.losses import DiceCELoss
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import SegmentationMeter
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the conditional diffusion segmentation model.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--data-config", default="src/configs/dataset.yaml", help="Path to dataset YAML config.")
    parser.add_argument("--binary", action="store_true", help="Train binary foreground segmentation.")
    return parser.parse_args()


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    loader,
    device: torch.device,
    num_classes: int,
    sample_steps: int,
) -> dict[str, float]:
    model.eval()
    meter = SegmentationMeter(num_classes)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = scheduler.ddim_sample(model, image, mask_channels=num_classes, sample_steps=sample_steps)
        pred = logits.argmax(dim=1)
        meter.update(pred, mask)
    return meter.compute()


def main() -> None:
    args = parse_args()
    exp_cfg, data_cfg = load_experiment_and_data(args.config, args.data_config)
    seed_everything(int(data_cfg.get("seed", 2026)))
    device = get_device()
    num_classes = 2 if args.binary else 4
    input_mode = exp_cfg.get("data", {}).get("input_mode", "2d")

    train_loader, val_loader = make_loaders(exp_cfg, data_cfg, binary=args.binary, input_mode=input_mode)
    model = build_experiment_model(exp_cfg, num_classes=num_classes).to(device)
    scheduler = DiffusionScheduler(
        timesteps=int(exp_cfg["diffusion"]["train_steps"]),
        beta_schedule=str(exp_cfg["diffusion"].get("beta_schedule", "cosine")),
    ).to(device)
    seg_criterion = DiceCELoss(num_classes=num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(exp_cfg["training"]["learning_rate"]),
        weight_decay=float(exp_cfg["training"].get("weight_decay", 0.0)),
    )
    amp = bool(exp_cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp)
    history_path = log_path(exp_cfg)
    loss_cfg = exp_cfg["training"]["loss"]
    noise_weight = float(loss_cfg.get("noise_mse_weight", 1.0))
    dice_ce_weight = float(loss_cfg.get("dice_ce_weight", 0.0))
    val_sample_steps = int(exp_cfg["evaluation"].get("validation_sample_steps", 25))
    best_dice = -1.0

    print(f"Training {exp_cfg['experiment_name']} on {device} with input_mode={input_mode}")
    for epoch in range(1, int(exp_cfg["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            mask_one_hot = mask_to_one_hot(mask, num_classes).to(device)
            t = torch.randint(0, scheduler.timesteps, (image.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(mask_one_hot)
            noisy_mask = scheduler.q_sample(mask_one_hot, t, noise)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                pred_noise = model(image, noisy_mask, t)
                noise_loss = F.mse_loss(pred_noise, noise)
                if dice_ce_weight:
                    pred_x0 = scheduler.predict_x0_from_noise(noisy_mask, t, pred_noise)
                    seg_loss = seg_criterion(pred_x0, mask)
                else:
                    seg_loss = torch.zeros((), device=device)
                loss = noise_weight * noise_loss + dice_ce_weight * seg_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * image.shape[0]
            total_items += image.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}", noise=f"{noise_loss.item():.4f}")

        train_loss = total_loss / max(total_items, 1)
        metrics = validate(model, scheduler, val_loader, device, num_classes, val_sample_steps)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        append_history(history_path, row)
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} mean_dice={metrics['mean_dice']:.4f}")
        if metrics["mean_dice"] > best_dice:
            best_dice = metrics["mean_dice"]
            save_checkpoint(
                checkpoint_path(exp_cfg),
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_mean_dice": best_dice,
                    "config": exp_cfg,
                    "binary": args.binary,
                },
            )
            print(f"  saved best checkpoint: mean_dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
