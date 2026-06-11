#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m src.engine.infer \
  --config src/configs/diffusion_x0_2d.yaml \
  --data-config src/configs/dataset.yaml \
  --output-dir results/figures/qualitative \
  --num-images 10 \
  --sample-steps 5 \
  --denoising \
  --uncertainty \
  --foreground-only \
  --min-foreground-pixels 1000 \
  --max-per-case 2
