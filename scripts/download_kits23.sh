#!/usr/bin/env bash
set -euo pipefail

python -m src.data.download_kits23_hf "$@"
