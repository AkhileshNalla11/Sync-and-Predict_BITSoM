#!/usr/bin/env bash
# One command to regenerate data and (in later stages) train + evaluate everything.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_python.sh

echo "[1/1] Generating synthetic EHR + wearable data"
"$PY" -m src.data.synthetic_gen --config config.yaml
# Later stages append: features -> baselines -> fusion training -> ablation
