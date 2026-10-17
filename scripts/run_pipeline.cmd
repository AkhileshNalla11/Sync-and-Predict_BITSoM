@echo off
rem One command to regenerate the data and train + evaluate every model (Windows, CPU, ~25 min).
rem Usage:  scripts\run_pipeline.cmd
setlocal
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo [1/5] Generating synthetic EHR + wearable data
"%PY%" -m src.data.synthetic_gen --config config.yaml || goto :fail

echo [2/5] Linking streams, patient-level split, windows + static features
"%PY%" -m src.data.build_dataset --config config.yaml || goto :fail

echo [3/5] Baselines: persistence + LightGBM (CGM / +wearables / +EHR)
"%PY%" -m src.models.baselines --config config.yaml || goto :fail

echo [4/5] Fusion model: GRU with EHR branch + sequence-only twin
"%PY%" -m src.models.train --config config.yaml --variants fusion_gru,seqonly_gru || goto :fail

echo [5/5] Ablation report: tables + figures in docs/
"%PY%" -m src.eval.ablation --config config.yaml || goto :fail

echo Done.
exit /b 0

:fail
echo Pipeline step failed (exit code %errorlevel%).
exit /b %errorlevel%
