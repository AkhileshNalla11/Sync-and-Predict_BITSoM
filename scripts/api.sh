#!/usr/bin/env bash
# Launch the FHIR risk API at http://localhost:8000 (interactive docs at /docs).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_python.sh
"$PY" -m uvicorn src.api.app:app --port 8000
