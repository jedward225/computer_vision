from __future__ import annotations

import torch


def confusion_counts(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.view(-1)
    target = target.view(-1)
    intersections = []
    pred_counts = []
    target_counts = []
    for cls in range(num_classes):
        pred_c = pred == cls
        target_c = target == cls
        intersections.append((pred_c & target_c).sum())
        pred_counts.append(pred_c.sum())
        target_counts.append(target_c.sum())
    return torch.stack(intersections), torch.stack(pred_counts), torch.stack(target_counts)


def dice_iou_from_counts(
    intersections: torch.Tensor,
    pred_counts: torch.Tensor,
    target_counts: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    dice = (2 * intersections.float() + eps) / (pred_counts.float() + target_counts.float() + eps)
    union = pred_counts.float() + target_counts.float() - intersections.float()
    iou = (intersections.float() + eps) / (union + eps)
    return dice, iou


class SegmentationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.intersections = torch.zeros(num_classes, dtype=torch.float64)
        self.pred_counts = torch.zeros(num_classes, dtype=torch.float64)
        self.target_counts = torch.zeros(num_classes, dtype=torch.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        inter, pred_c, target_c = confusion_counts(pred.cpu(), target.cpu(), self.num_classes)
        self.intersections += inter.double()
        self.pred_counts += pred_c.double()
        self.target_counts += target_c.double()

    def compute(self) -> dict[str, float]:
        dice, iou = dice_iou_from_counts(self.intersections, self.pred_counts, self.target_counts)
        foreground = slice(1, self.num_classes) if self.num_classes > 1 else slice(0, self.num_classes)
        return {
            "mean_dice": float(dice[foreground].mean().item()),
            "mean_iou": float(iou[foreground].mean().item()),
            **{f"dice_class_{idx}": float(dice[idx].item()) for idx in range(self.num_classes)},
            **{f"iou_class_{idx}": float(iou[idx].item()) for idx in range(self.num_classes)},
        }

