#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/data/work/MichaelYu"
PORT="${PORT:-8092}"
PYTHON_BIN="${PYTHON_BIN:-/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python}"
RESULT_FILE="${WORKSPACE_ROOT}/florence-caption/multi_region_description/eval/results/qwen_person_2to5_loc_gpu2_ep2_bs16_lr8e6_fixed.jsonl"
PAGE_PATH="/florence-caption/multi_region_description/eval/viewer/index.html"

if [[ ! -f "${RESULT_FILE}" ]]; then
    echo "Evaluation result not found: ${RESULT_FILE}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

echo "Serving workspace: ${WORKSPACE_ROOT}"
echo "Viewer URL       : http://127.0.0.1:${PORT}${PAGE_PATH}"
echo "Press Ctrl+C to stop."

cd "${WORKSPACE_ROOT}"
exec "${PYTHON_BIN}" -m http.server "${PORT}" --bind 0.0.0.0
