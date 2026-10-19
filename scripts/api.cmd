@echo off
rem Launch the FHIR risk API (Windows: PowerShell or Command Prompt).
rem Usage:  scripts\api.cmd      (opens http://localhost:8000/docs; Ctrl+C to stop)
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start "" http://localhost:8000/docs"
"%PY%" -m uvicorn src.api.app:app --port 8000
