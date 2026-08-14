#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/data/work/MichaelYu"
VIEWER_DIR="${WORKSPACE_ROOT}/florence-caption/multi_region_description/eval/assignment_badcase_viewer"
PYTHON_BIN="${PYTHON_BIN:-/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python}"
PORT="${PORT:-8093}"
PAGE_PATH="/florence-caption/multi_region_description/eval/assignment_badcase_viewer/index.html"

"${PYTHON_BIN}" "${VIEWER_DIR}/build_badcases.py"

echo "Badcase viewer URL: http://127.0.0.1:${PORT}${PAGE_PATH}"
echo "Press Ctrl+C to stop."

cd "${WORKSPACE_ROOT}"
exec "${PYTHON_BIN}" -m http.server "${PORT}" --bind 0.0.0.0
