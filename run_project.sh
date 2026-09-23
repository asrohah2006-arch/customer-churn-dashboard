#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PY=""
if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
elif python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; then
  PY=python3
else
  echo "This project needs Python 3.11 (TensorFlow does not support newer versions yet)."
  echo "Install Python 3.11 from https://www.python.org/downloads/ and run this script again."
  exit 1
fi

if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; then
  echo "Removing .venv created with a different Python version..."
  rm -rf .venv
fi
[ -x .venv/bin/python ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary :all: -r requirements.txt
python src/eda.py
python src/train.py --epochs 80
streamlit run app.py
