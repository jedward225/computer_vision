#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m src.engine.infer \
  --config src/configs/diffusion_2d.yaml \
  --data-config src/configs/dataset.yaml \
  --output-dir results/figures/qualitative \
  --num-images 10 \
  --sample-steps 25 \
  --denoising \
  --uncertainty
