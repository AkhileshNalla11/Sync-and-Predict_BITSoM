# Sourced by the other scripts: pick the repo virtualenv's Python if present.
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY="${PYTHON:-python}"; fi
