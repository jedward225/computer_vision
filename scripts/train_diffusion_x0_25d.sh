#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m src.engine.train_diffusion --config src/configs/diffusion_x0_25d.yaml --data-config src/configs/dataset.yaml
