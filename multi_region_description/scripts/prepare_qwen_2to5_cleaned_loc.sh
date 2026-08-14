#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_ROOT}/miniconda3/envs/florence-caption/bin/python}"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_data.py"
INPUT_DIR="${INPUT_DIR:-${WORKSPACE_ROOT}/florence-data/qwen_person_2to6_region_descriptions_qwen_cleaned}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/multi_region_description/data/qwen-person-2to5-cleaned-description-loc}"

FORCE=false
CHECK_ONLY=false

usage() {
    cat <<EOF
Usage: $0 [--output-dir DIR] [--force] [--check-only]

Rebuild the dataset used by:
  multi_region_description/checkpoints/qwen_person_2to5_cleaned_loc_gpu1_ep2_bs16_lr8e6

Environment overrides:
  PYTHON_BIN  Python interpreter
  INPUT_DIR   Cleaned per-frame Qwen JSON root
  OUTPUT_DIR  Output directory for train.jsonl and test.jsonl
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --output-dir)
            if (( $# < 2 )); then
                echo "--output-dir requires a directory." >&2
                exit 2
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for required_file in "${PYTHON_BIN}" "${PREPARE_SCRIPT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "Input directory not found: ${INPUT_DIR}" >&2
    exit 1
fi

echo "Input       : ${INPUT_DIR}"
echo "Output      : ${OUTPUT_DIR}"
echo "Description : qwen.description with status=success"
echo "Crop counts : 2 3 4 5"
echo "Prompt      : <REGIONS_TO_DESCRIPTIONS><loc...><sep><loc...>"
echo "Label       : description<loc_x1><loc_y1><loc_x2><loc_y2>..."
echo "Split       : test_ratio=0.10 seed=42"

if [[ "${CHECK_ONLY}" == true ]]; then
    echo "Check-only mode: inputs are available; no files were written."
    exit 0
fi

TRAIN_FILE="${OUTPUT_DIR}/train.jsonl"
TEST_FILE="${OUTPUT_DIR}/test.jsonl"
if [[ "${FORCE}" != true && ( -e "${TRAIN_FILE}" || -e "${TEST_FILE}" ) ]]; then
    echo "Refusing to overwrite existing train/test files in ${OUTPUT_DIR}." >&2
    echo "Use --force or choose --output-dir DIR." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
if [[ "${FORCE}" == true ]]; then
    rm -f -- "${TRAIN_FILE}" "${TEST_FILE}"
fi

"${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --manifest-file "" \
    --label-format loc \
    --allowed-crop-counts 2 3 4 5 \
    --description-source qwen \
    --test-ratio 0.10 \
    --seed 42

"${PYTHON_BIN}" - "${TRAIN_FILE}" "${TEST_FILE}" <<'PY'
import json
import re
import sys
from pathlib import Path

task_token = "<REGIONS_TO_DESCRIPTIONS>"
loc_pattern = re.compile(r"<loc_\d+><loc_\d+><loc_\d+><loc_\d+>")

for path_text in sys.argv[1:]:
    path = Path(path_text)
    rows = 0
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            sample = json.loads(line)
            prompt = sample.get("prompt", "")
            label = sample.get("label", "")
            if not prompt.startswith(task_token):
                raise ValueError(f"{path}:{line_number}: invalid task token")
            prompt_boxes = prompt[len(task_token):].split("<sep>")
            if not 2 <= len(prompt_boxes) <= 5:
                raise ValueError(f"{path}:{line_number}: crop count is not in [2, 5]")
            if any(loc_pattern.fullmatch(box) is None for box in prompt_boxes):
                raise ValueError(f"{path}:{line_number}: malformed prompt bbox")
            label_boxes = loc_pattern.findall(label)
            if label_boxes != prompt_boxes:
                raise ValueError(f"{path}:{line_number}: prompt/label bbox order differs")
            rows += 1
    print(f"Validated {rows} rows: {path}")
PY
