from __future__ import annotations

import argparse

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.engine.common import (
    append_history,
    build_experiment_model,
    checkpoint_path,
    get_device,
    load_experiment_and_data,
    log_path,
    make_loaders,
)
from src.engine.losses import DiceCELoss, kl_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import SegmentationMeter
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the conditional VAE segmentation baseline.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--data-config", default="src/configs/dataset.yaml", help="Path to dataset YAML config.")
    parser.add_argument("--binary", action="store_true", help="Train binary foreground segmentation.")
    return parser.parse_args()


@torch.no_grad()
def validate(model: torch.nn.Module, loader, criterion: DiceCELoss, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    meter = SegmentationMeter(num_classes)
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        outputs = model(image, mask)
        loss = criterion(outputs["logits"], mask)
        pred = outputs["logits"].argmax(dim=1)
        meter.update(pred, mask)
        total_loss += float(loss.item()) * image.shape[0]
        total_items += image.shape[0]
    metrics = meter.compute()
    metrics["val_loss"] = total_loss / max(total_items, 1)
    return metrics


def main() -> None:
    args = parse_args()
    exp_cfg, data_cfg = load_experiment_and_data(args.config, args.data_config)
    seed_everything(int(data_cfg.get("seed", 2026)))
    device = get_device()
    num_classes = 2 if args.binary else 4

    train_loader, val_loader = make_loaders(exp_cfg, data_cfg, binary=args.binary, input_mode="2d")
    model = build_experiment_model(exp_cfg, num_classes=num_classes).to(device)
    criterion = DiceCELoss(num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(exp_cfg["training"]["learning_rate"]))
    beta_kl = float(exp_cfg["training"].get("beta_kl", 1e-3))
    amp = bool(exp_cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp)
    history_path = log_path(exp_cfg)
    best_dice = -1.0

    print(f"Training {exp_cfg['experiment_name']} on {device}")
    for epoch in range(1, int(exp_cfg["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                outputs = model(image, mask)
                recon_loss = criterion(outputs["logits"], mask)
                loss = recon_loss + beta_kl * kl_loss(outputs["mu"], outputs["logvar"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * image.shape[0]
            total_items += image.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(total_items, 1)
        metrics = validate(model, val_loader, criterion, device, num_classes)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        append_history(history_path, row)
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} mean_dice={metrics['mean_dice']:.4f}"
        )
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
