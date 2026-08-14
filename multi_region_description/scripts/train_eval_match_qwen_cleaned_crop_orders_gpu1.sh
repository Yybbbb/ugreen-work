#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/work/MichaelYu/florence-caption}"
PYTHON_BIN="${PYTHON_BIN:-/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python}"
GPU_INDEX="${GPU_INDEX:-1}"
MIN_FREE_MEMORY_MIB="${MIN_FREE_MEMORY_MIB:-60000}"

BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/ugipc_1231_15words_epoch3_full_handoff/checkpoint}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/multi_region_description/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/multi_region_description/checkpoints}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/multi_region_description/logs}"
EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT:-${PROJECT_ROOT}/multi_region_description/eval/results}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${PROJECT_ROOT}/multi_region_description/eval/logs}"
MATCHING_ROOT="${MATCHING_ROOT:-${EVAL_RESULT_ROOT}/description_matching_cleaned_crop_orders}"

TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_multi_region.py"
EVAL_SCRIPT="${PROJECT_ROOT}/multi_region_description/eval/scripts/eval_plan_b.py"
MATCHING_SCRIPT="${PROJECT_ROOT}/multi_region_description/eval/scripts/evaluate_description_matching.py"
RUN_PREFIX="${RUN_PREFIX:-qwen_person_2to5_cleaned_loc_order_gpu1_ep2_bs16_lr8e6}"

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-8e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_STEPS="${LOG_STEPS:-10}"
SEED="${SEED:-42}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-768}"
TARGET_MAX_LENGTH="${TARGET_MAX_LENGTH:-320}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
NUM_BEAMS="${NUM_BEAMS:-1}"

declare -a ORDERS=("area_desc" "center_left_to_right")

dataset_dir_for() {
    case "$1" in
        area_desc) echo "qwen-person-2to5-cleaned-description-loc-order-area-desc" ;;
        center_left_to_right) echo "qwen-person-2to5-cleaned-description-loc-order-center-left-to-right" ;;
        *) echo "Unsupported crop order: $1" >&2; return 1 ;;
    esac
}

require_file() {
    [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }
}

require_dir() {
    [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }
}

require_file "${PYTHON_BIN}"
require_file "${TRAIN_SCRIPT}"
require_file "${EVAL_SCRIPT}"
require_file "${MATCHING_SCRIPT}"
require_dir "${BASE_MODEL}"

for order in "${ORDERS[@]}"; do
    dataset_dir="$(dataset_dir_for "${order}")"
    require_file "${DATA_ROOT}/${dataset_dir}/train.jsonl"
    require_file "${DATA_ROOT}/${dataset_dir}/test.jsonl"
    run_name="${RUN_PREFIX}_${order}"
    matching_dir="${MATCHING_ROOT}/${order}"
    if [[ -e "${CHECKPOINT_ROOT}/${run_name}" \
        || -e "${LOG_ROOT}/${run_name}" \
        || -e "${EVAL_LOG_ROOT}/${run_name}" \
        || -e "${EVAL_RESULT_ROOT}/${run_name}.jsonl" \
        || -e "${matching_dir}" ]]; then
        echo "Refusing to overwrite existing output for run: ${run_name}" >&2
        exit 1
    fi
done

free_memory_mib="$(
    nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.free \
        --format=csv,noheader,nounits | tr -d '[:space:]'
)"
if [[ ! "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < MIN_FREE_MEMORY_MIB )); then
    echo "GPU ${GPU_INDEX} free memory ${free_memory_mib:-unknown} MiB is below ${MIN_FREE_MEMORY_MIB} MiB." >&2
    exit 1
fi

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "${EVAL_RESULT_ROOT}" "${EVAL_LOG_ROOT}" "${MATCHING_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export TOKENIZERS_PARALLELISM=false

echo "Base model       : ${BASE_MODEL}"
echo "GPU              : physical ${GPU_INDEX} -> cuda:0"
echo "GPU free memory  : ${free_memory_mib} MiB"
echo "Epochs/batch/lr  : ${EPOCHS}/${BATCH_SIZE}/${LEARNING_RATE}"
echo "Target/generation: ${TARGET_MAX_LENGTH}/${MAX_NEW_TOKENS}"
echo "Runs             : ${ORDERS[*]}"

for order in "${ORDERS[@]}"; do
    dataset_dir="$(dataset_dir_for "${order}")"
    train_file="${DATA_ROOT}/${dataset_dir}/train.jsonl"
    test_file="${DATA_ROOT}/${dataset_dir}/test.jsonl"
    run_name="${RUN_PREFIX}_${order}"
    final_checkpoint="${CHECKPOINT_ROOT}/${run_name}/final"
    eval_output="${EVAL_RESULT_ROOT}/${run_name}.jsonl"
    eval_log_dir="${EVAL_LOG_ROOT}/${run_name}"
    matching_dir="${MATCHING_ROOT}/${order}"

    mkdir -p "${eval_log_dir}" "${matching_dir}"
    echo "============================================================"
    echo "Starting ${order}: ${run_name}"
    echo "============================================================"

    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
        --data "${train_file}" \
        --model-path "${BASE_MODEL}" \
        --output-root "${CHECKPOINT_ROOT}" \
        --log-root "${LOG_ROOT}" \
        --run-name "${run_name}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LEARNING_RATE}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --precision "${PRECISION}" \
        --num-workers "${NUM_WORKERS}" \
        --log-steps "${LOG_STEPS}" \
        --seed "${SEED}" \
        --prompt-max-length "${PROMPT_MAX_LENGTH}" \
        --target-max-length "${TARGET_MAX_LENGTH}"

    require_dir "${final_checkpoint}"

    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --checkpoint "${final_checkpoint}" \
        --test-file "${test_file}" \
        --output-file "${eval_output}" \
        --device cuda:0 \
        --dtype "${PRECISION}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --num-beams "${NUM_BEAMS}" \
        2>&1 | tee "${eval_log_dir}/eval.log"

    "${PYTHON_BIN}" - "${eval_output}" "${matching_dir}" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
matched_path = output_dir / "count_matched_input.jsonl"
mismatched_path = output_dir / "count_mismatched_records.jsonl"
matched = []
mismatched = []
for line in source_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    destination = matched if len(record["gt_pairs"]) == len(record["pred_pairs"]) else mismatched
    destination.append(record)
for path, records in ((matched_path, matched), (mismatched_path, mismatched)):
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"description matching input: matched={len(matched)} mismatched={len(mismatched)}")
PY

    "${PYTHON_BIN}" "${MATCHING_SCRIPT}" \
        --input "${matching_dir}/count_matched_input.jsonl" \
        --output-dir "${matching_dir}" \
        2>&1 | tee "${matching_dir}/description_matching.log"

    echo "Completed ${order}"
    echo "Checkpoint : ${final_checkpoint}"
    echo "Evaluation : ${eval_output}"
    echo "Matching   : ${matching_dir}/description_matching_summary.json"
done

echo "Both cleaned crop-order experiments completed."
