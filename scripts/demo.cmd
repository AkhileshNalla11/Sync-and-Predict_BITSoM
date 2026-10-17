@echo off
rem Launch the clinician dashboard (Windows: PowerShell or Command Prompt).
rem Usage:  scripts\demo.cmd      then open http://localhost:8501
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -m streamlit run src/app/dashboard.py
