#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m src.engine.evaluate --configs \
  src/configs/unet.yaml \
  src/configs/unet_25d.yaml \
  src/configs/segresnet.yaml \
  src/configs/segresnet_25d.yaml \
  src/configs/cvae.yaml \
  src/configs/diffusion_x0_2d.yaml \
  --data-config src/configs/dataset.yaml \
  --output results/tables/evaluation_axial_extended.csv
