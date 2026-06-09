#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m src.engine.train_diffusion --config src/configs/diffusion_2d.yaml --data-config src/configs/dataset.yaml
