from __future__ import annotations

from pathlib import Path

import numpy as np


PALETTE = np.array(
    [
        [0, 0, 0],
        [56, 160, 115],
        [213, 94, 0],
        [0, 114, 178],
    ],
    dtype=np.float32,
)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(np.int64)
    rgb = PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]
    return rgb.astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[0]
    image = np.clip(image, 0, 1)
    gray = np.repeat((image[..., None] * 255), 3, axis=-1)
    color = mask_to_rgb(mask).astype(np.float32)
    foreground = (mask > 0)[..., None]
    out = np.where(foreground, (1 - alpha) * gray + alpha * color, gray)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_panel(path: str | Path, images: list[np.ndarray], titles: list[str] | None = None) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for idx, (ax, image) in enumerate(zip(axes, images, strict=True)):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)
        if titles:
            ax.set_title(titles[idx])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

