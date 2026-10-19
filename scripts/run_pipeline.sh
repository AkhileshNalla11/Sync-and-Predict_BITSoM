#!/usr/bin/env bash
# One command to regenerate the data and train + evaluate every model (CPU, ~25 min).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_python.sh

echo "[1/9] Generating synthetic EHR + wearable data"
"$PY" -m src.data.synthetic_gen --config config.yaml

echo "[2/9] Linking streams, patient-level split, windows + static features"
"$PY" -m src.data.build_dataset --config config.yaml

echo "[3/9] Baselines: persistence + LightGBM (CGM / +wearables / +EHR)"
"$PY" -m src.models.baselines --config config.yaml

echo "[4/9] Fusion model: GRU with EHR branch + sequence-only twin"
"$PY" -m src.models.train --config config.yaml --variants fusion_gru,seqonly_gru

echo "[5/9] Ablation report: tables + figures in docs/"
"$PY" -m src.eval.ablation --config config.yaml

echo "[6/9] Personalised twin: fit test patients, held-out validation report"
"$PY" -m src.eval.twin_validation --config config.yaml

echo "[7/9] Twin uncertainty bands: calibrate on validation patients, coverage on test"
"$PY" -m src.eval.twin_intervals --config config.yaml

echo "[8/9] Alert policy: choose operating points on validation patients, episode-level test report"
"$PY" -m src.eval.alert_policy --config config.yaml

echo "[9/9] Subgroup performance of the deployed model"
"$PY" -m src.eval.subgroups --config config.yaml
