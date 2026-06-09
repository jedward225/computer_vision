#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m src.engine.train_cvae --config src/configs/cvae.yaml --data-config src/configs/dataset.yaml
