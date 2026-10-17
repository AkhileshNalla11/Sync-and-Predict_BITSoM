@echo off
rem Launch the clinician dashboard (Windows: PowerShell or Command Prompt).
rem Usage:  scripts\demo.cmd      (opens http://localhost:8501 in your browser; Ctrl+C to stop)
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
rem Open the browser a few seconds after the server starts (Streamlit itself runs headless).
start "" /b cmd /c "timeout /t 5 /nobreak >nul & start "" http://localhost:8501"
"%PY%" -m streamlit run src/app/dashboard.py
