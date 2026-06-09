#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m src.engine.evaluate --configs \
  src/configs/unet.yaml \
  src/configs/segresnet.yaml \
  src/configs/cvae.yaml \
  src/configs/diffusion_2d.yaml \
  src/configs/diffusion_25d.yaml \
  --data-config src/configs/dataset.yaml
