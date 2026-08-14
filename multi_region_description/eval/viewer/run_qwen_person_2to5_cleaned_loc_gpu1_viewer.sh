#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/data/work/MichaelYu"
PORT="${PORT:-8093}"
PYTHON_BIN="${PYTHON_BIN:-/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python}"
RESULT_FILE="${WORKSPACE_ROOT}/florence-caption/multi_region_description/eval/results/qwen_person_2to5_cleaned_loc_gpu1_ep2_bs16_lr8e6.jsonl"
ATTRIBUTE_SUMMARY="${WORKSPACE_ROOT}/florence-caption/multi_region_description/eval/results/qwen_person_2to5_cleaned_loc_gpu1_ep2_bs16_lr8e6_analysis/attribute_count_distribution.json"
ATTRIBUTE_DETAILS="${WORKSPACE_ROOT}/florence-caption/multi_region_description/eval/results/qwen_person_2to5_cleaned_loc_gpu1_ep2_bs16_lr8e6_analysis/person_attribute_counts.jsonl"
MATCHING_SUMMARY="${WORKSPACE_ROOT}/florence-caption/multi_region_description/eval/results/qwen_person_2to5_cleaned_loc_gpu1_ep2_bs16_lr8e6_analysis/description_matching/description_matching_summary.json"
MATCHING_DETAILS="${WORKSPACE_ROOT}/florence-caption/multi_region_description/eval/results/qwen_person_2to5_cleaned_loc_gpu1_ep2_bs16_lr8e6_analysis/description_matching/description_matching_results.jsonl"
PAGE_PATH="/florence-caption/multi_region_description/eval/viewer/qwen_person_2to5_cleaned_loc_gpu1.html"

for required_file in "${RESULT_FILE}" "${ATTRIBUTE_SUMMARY}" "${ATTRIBUTE_DETAILS}" "${MATCHING_SUMMARY}" "${MATCHING_DETAILS}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Viewer input not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

echo "Serving workspace: ${WORKSPACE_ROOT}"
echo "Local URL        : http://127.0.0.1:${PORT}${PAGE_PATH}"
for host_ip in $(hostname -I); do
    echo "Server URL       : http://${host_ip}:${PORT}${PAGE_PATH}"
done
echo "SSH tunnel       : ssh -L ${PORT}:127.0.0.1:${PORT} <user>@$(hostname)"
echo "Press Ctrl+C to stop."

cd "${WORKSPACE_ROOT}"
exec "${PYTHON_BIN}" -m http.server "${PORT}" --bind 0.0.0.0
