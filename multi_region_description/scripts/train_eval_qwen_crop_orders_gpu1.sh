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

TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_multi_region.py"
EVAL_SCRIPT="${PROJECT_ROOT}/multi_region_description/eval/scripts/eval_plan_b.py"
RUN_PREFIX="${RUN_PREFIX:-qwen_person_2to5_loc_order_gpu1_ep2_bs16_lr8e6}"

# Keep the same hyperparameters as the previous qwen-person description+loc run.
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
TARGET_MAX_LENGTH="${TARGET_MAX_LENGTH:-160}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
NUM_BEAMS="${NUM_BEAMS:-1}"

declare -a ORDERS=("random" "area_desc" "center_left_to_right")

dataset_dir_for() {
    case "$1" in
        random) echo "qwen-person-2to5-description-loc-order-random" ;;
        area_desc) echo "qwen-person-2to5-description-loc-order-area-desc" ;;
        center_left_to_right) echo "qwen-person-2to5-description-loc-order-center-left-to-right" ;;
        *) echo "Unsupported crop order: $1" >&2; return 1 ;;
    esac
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory not found: $1" >&2
        exit 1
    fi
}

require_file "${PYTHON_BIN}"
require_file "${TRAIN_SCRIPT}"
require_file "${EVAL_SCRIPT}"
require_dir "${BASE_MODEL}"

for order in "${ORDERS[@]}"; do
    dataset_dir="$(dataset_dir_for "${order}")"
    require_file "${DATA_ROOT}/${dataset_dir}/train.jsonl"
    require_file "${DATA_ROOT}/${dataset_dir}/test.jsonl"

    run_name="${RUN_PREFIX}_${order}"
    if [[ -e "${CHECKPOINT_ROOT}/${run_name}" \
        || -e "${LOG_ROOT}/${run_name}" \
        || -e "${EVAL_LOG_ROOT}/${run_name}" \
        || -e "${EVAL_RESULT_ROOT}/${run_name}.jsonl" ]]; then
        echo "Refusing to overwrite existing output for run: ${run_name}" >&2
        echo "Set RUN_PREFIX to a new value or remove the existing outputs." >&2
        exit 1
    fi
done

free_memory_mib="$(
    nvidia-smi \
        --id="${GPU_INDEX}" \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits | tr -d '[:space:]'
)"

if [[ ! "${free_memory_mib}" =~ ^[0-9]+$ ]]; then
    echo "Unable to read free memory for GPU ${GPU_INDEX}." >&2
    exit 1
fi
if (( free_memory_mib < MIN_FREE_MEMORY_MIB )); then
    echo "GPU ${GPU_INDEX} has only ${free_memory_mib} MiB free; ${MIN_FREE_MEMORY_MIB} MiB is required." >&2
    exit 1
fi

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "${EVAL_RESULT_ROOT}" "${EVAL_LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export TOKENIZERS_PARALLELISM=false

echo "Base model       : ${BASE_MODEL}"
echo "GPU              : physical ${GPU_INDEX} -> cuda:0"
echo "GPU free memory  : ${free_memory_mib} MiB"
echo "Epochs/batch/lr  : ${EPOCHS}/${BATCH_SIZE}/${LEARNING_RATE}"
echo "Warmup/precision : ${WARMUP_STEPS}/${PRECISION}"
echo "Runs             : ${ORDERS[*]}"
echo

for order in "${ORDERS[@]}"; do
    dataset_dir="$(dataset_dir_for "${order}")"
    train_file="${DATA_ROOT}/${dataset_dir}/train.jsonl"
    test_file="${DATA_ROOT}/${dataset_dir}/test.jsonl"
    run_name="${RUN_PREFIX}_${order}"
    final_checkpoint="${CHECKPOINT_ROOT}/${run_name}/final"
    eval_output="${EVAL_RESULT_ROOT}/${run_name}.jsonl"
    eval_log_dir="${EVAL_LOG_ROOT}/${run_name}"

    mkdir -p "${eval_log_dir}"

    echo "============================================================"
    echo "Crop order       : ${order}"
    echo "Run name         : ${run_name}"
    echo "Train data       : ${train_file}"
    echo "Test data        : ${test_file}"
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

    echo "Training completed. Evaluating ${run_name}."
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --checkpoint "${final_checkpoint}" \
        --test-file "${test_file}" \
        --output-file "${eval_output}" \
        --device cuda:0 \
        --dtype "${PRECISION}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --num-beams "${NUM_BEAMS}" \
        2>&1 | tee "${eval_log_dir}/eval.log"

    echo "Completed         : ${run_name}"
    echo "Checkpoint        : ${final_checkpoint}"
    echo "Evaluation result : ${eval_output}"
    echo
done

echo "All three crop-order training and evaluation runs completed."
