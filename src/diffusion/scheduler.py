from __future__ import annotations

import math

import torch
from torch import nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class DiffusionScheduler(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "cosine") -> None:
        super().__init__()
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def predict_x0_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        return (x_t - sqrt_one_minus * noise) / torch.clamp(sqrt_alpha, min=1e-8)

    def predict_noise_from_x0(self, x_t: torch.Tensor, timesteps: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape)
        return (x_t - sqrt_alpha * x0) / torch.clamp(sqrt_one_minus, min=1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        image: torch.Tensor,
        mask_channels: int,
        sample_steps: int = 25,
        return_trajectory: bool = False,
        clip_x0: bool = True,
        prediction_type: str = "epsilon",
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        device = image.device
        b, _, h, w = image.shape
        x = torch.randn(b, mask_channels, h, w, device=device)
        timesteps = torch.linspace(self.timesteps - 1, 0, sample_steps, device=device).long()
        trajectory: list[torch.Tensor] = []

        for idx, t in enumerate(timesteps):
            t_batch = torch.full((b,), int(t.item()), device=device, dtype=torch.long)
            model_out = model(image, x, t_batch)
            if prediction_type == "epsilon":
                pred_noise = model_out
                x0 = self.predict_x0_from_noise(x, t_batch, pred_noise)
                if clip_x0:
                    x0 = torch.nan_to_num(x0, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            elif prediction_type == "x0":
                x0 = model_out.softmax(dim=1)
                pred_noise = self.predict_noise_from_x0(x, t_batch, x0)
            else:
                raise ValueError(f"Unknown prediction_type: {prediction_type}")

            if return_trajectory:
                trajectory.append(x0.detach().cpu())

            if idx == len(timesteps) - 1:
                x = x0
                break

            t_prev = timesteps[idx + 1]
            alpha_prev = self.alphas_cumprod[int(t_prev.item())]
            x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1 - alpha_prev) * pred_noise

        if return_trajectory:
            return x, trajectory
        return x
