#!/usr/bin/env bash
# One command to regenerate the data and train + evaluate every model (CPU, ~25 min).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_python.sh

echo "[1/4] Generating synthetic EHR + wearable data"
"$PY" -m src.data.synthetic_gen --config config.yaml

echo "[2/4] Linking streams, patient-level split, windows + static features"
"$PY" -m src.data.build_dataset --config config.yaml

echo "[3/4] Baselines: persistence + LightGBM (CGM / +wearables / +EHR)"
"$PY" -m src.models.baselines --config config.yaml

echo "[4/4] Fusion model: GRU with EHR branch + sequence-only twin"
"$PY" -m src.models.train --config config.yaml --variants fusion_gru,seqonly_gru
