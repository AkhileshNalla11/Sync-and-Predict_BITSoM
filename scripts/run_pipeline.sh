#!/usr/bin/env bash
# One command to regenerate data and (in later stages) train + evaluate everything.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_python.sh

echo "[1/2] Generating synthetic EHR + wearable data"
"$PY" -m src.data.synthetic_gen --config config.yaml

echo "[2/2] Linking streams, patient-level split, windows + static features"
"$PY" -m src.data.build_dataset --config config.yaml
# Later stages append: features -> baselines -> fusion training -> ablation
