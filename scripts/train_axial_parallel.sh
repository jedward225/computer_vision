#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p logs/run

CUDA_VISIBLE_DEVICES=0 bash scripts/train_unet.sh > logs/run/unet_axial_2d.out 2>&1 &
pids=("$!")

CUDA_VISIBLE_DEVICES=1 bash scripts/train_segresnet.sh > logs/run/segresnet_axial_2d.out 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=2 bash scripts/train_cvae.sh > logs/run/cvae_axial_2d.out 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=3 bash scripts/train_diffusion_x0.sh > logs/run/diffusion_axial_2d_x0.out 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=4 python -m src.engine.train_baseline \
  --config src/configs/unet_25d.yaml \
  --data-config src/configs/dataset.yaml > logs/run/unet_axial_25d.out 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=5 python -m src.engine.train_baseline \
  --config src/configs/segresnet_25d.yaml \
  --data-config src/configs/dataset.yaml > logs/run/segresnet_axial_25d.out 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
