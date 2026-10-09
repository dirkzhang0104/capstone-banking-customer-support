#!/usr/bin/env bash
# One-shot launcher: creates a venv, installs deps, starts the Streamlit UI.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

.venv/bin/pip install -q -r requirements.txt
echo "Starting Streamlit UI on http://localhost:8501 ..."
exec .venv/bin/streamlit run app/main.py
