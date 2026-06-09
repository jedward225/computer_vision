#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python -m src.engine.train_baseline --config src/configs/segresnet.yaml --data-config src/configs/dataset.yaml
