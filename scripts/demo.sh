#!/usr/bin/env bash
# Launch the clinician dashboard (implemented in Stage 7).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_python.sh
"$PY" -m streamlit run src/app/dashboard.py
