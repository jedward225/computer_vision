#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m src.engine.make_report_figures \
  --output-dir results/figures \
  --experiments \
  unet_axial_2d_multiclass \
  segresnet_axial_2d_multiclass \
  cvae_axial_2d_multiclass \
  diffusion_axial_2d_x0_multiclass \
  unet_axial_25d_multiclass \
  segresnet_axial_25d_multiclass
