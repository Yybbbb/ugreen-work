#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/data1/work/MichaelYu/miniconda3/envs/florence/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8012}"

cd "${PROJECT_DIR}"

# Build the HTML + sample manifest if missing or when --rebuild is passed.
if [[ "${REBUILD:-1}" == "1" ]] || [[ ! -f "${SCRIPT_DIR}/build/index.html" ]]; then
  echo "[webdemo] building demo data ..."
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_demo.py"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
echo "[webdemo] serving on http://${HOST}:${PORT}  (open http://127.0.0.1:${PORT})"
exec "${PYTHON_BIN}" -m webdemo.server --host "${HOST}" --port "${PORT}"
