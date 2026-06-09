#!/usr/bin/env bash
set -euo pipefail

conda env create -f environment.yml
conda run -n cv-final python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

