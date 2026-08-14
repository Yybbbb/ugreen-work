#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/work/MichaelYu/florence-caption}"
PYTHON_BIN="${PYTHON_BIN:-/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_PREFIX="${RUN_PREFIX:-qwen_person_2to5_v1}"

BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/ugipc_1231_15words_epoch3_full_handoff/checkpoint}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/multi_region_description/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/multi_region_description/checkpoints}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/multi_region_description/logs}"
EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT:-${PROJECT_ROOT}/multi_region_description/eval/results}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${PROJECT_ROOT}/multi_region_description/eval/logs}"

TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_multi_region.py"
EVAL_SEP_SCRIPT="${PROJECT_ROOT}/multi_region_description/eval/scripts/eval.py"
EVAL_LOC_SCRIPT="${PROJECT_ROOT}/multi_region_description/eval/scripts/eval_plan_b.py"

# The source checkpoint is already specialized for short person descriptions.
# Use a conservative LR and two epochs to learn the multi-region grammar without
# unnecessarily drifting from the existing description capability.
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-8e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_STEPS="${LOG_STEPS:-10}"
SEED="${SEED:-42}"

# The measured maxima are 37 prompt tokens and 132 target tokens. Florence-2
# internally reserves 577 image tokens from prompt_max_length, so 768 leaves
# ample prompt space while 160 covers every current target without truncation.
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-768}"
TARGET_MAX_LENGTH="${TARGET_MAX_LENGTH:-160}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
NUM_BEAMS="${NUM_BEAMS:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false

declare -a FORMATS=("sep" "loc" "loc-sep")

dataset_dir_for() {
    case "$1" in
        sep) echo "qwen-person-2to5-description-sep" ;;
        loc) echo "qwen-person-2to5-description-loc" ;;
        loc-sep) echo "qwen-person-2to5-description-loc-sep" ;;
        *) echo "Unsupported format: $1" >&2; return 1 ;;
    esac
}

eval_script_for() {
    case "$1" in
        sep) echo "${EVAL_SEP_SCRIPT}" ;;
        loc|loc-sep) echo "${EVAL_LOC_SCRIPT}" ;;
        *) echo "Unsupported format: $1" >&2; return 1 ;;
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
require_file "${EVAL_SEP_SCRIPT}"
require_file "${EVAL_LOC_SCRIPT}"
require_dir "${BASE_MODEL}"

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "${EVAL_RESULT_ROOT}" "${EVAL_LOG_ROOT}"

echo "Base model       : ${BASE_MODEL}"
echo "GPU              : physical ${GPU_ID} -> cuda:0"
echo "Precision        : ${PRECISION}"
echo "Epochs           : ${EPOCHS}"
echo "Batch size       : ${BATCH_SIZE}"
echo "Learning rate    : ${LEARNING_RATE}"
echo "Warmup steps     : ${WARMUP_STEPS}"
echo "Seed             : ${SEED}"
echo "Prompt/target max: ${PROMPT_MAX_LENGTH}/${TARGET_MAX_LENGTH}"
echo

for format in "${FORMATS[@]}"; do
    dataset_dir="$(dataset_dir_for "${format}")"
    train_file="${DATA_ROOT}/${dataset_dir}/train.jsonl"
    test_file="${DATA_ROOT}/${dataset_dir}/test.jsonl"
    run_name="${RUN_PREFIX}_${format//-/_}"
    final_checkpoint="${CHECKPOINT_ROOT}/${run_name}/final"
    train_log_dir="${LOG_ROOT}/${run_name}"
    eval_log_dir="${EVAL_LOG_ROOT}/${run_name}"
    eval_output="${EVAL_RESULT_ROOT}/${run_name}.jsonl"
    eval_script="$(eval_script_for "${format}")"

    require_file "${train_file}"
    require_file "${test_file}"

    if [[ -e "${CHECKPOINT_ROOT}/${run_name}" || -e "${train_log_dir}" || -e "${eval_log_dir}" || -e "${eval_output}" ]]; then
        echo "Refusing to overwrite existing output for run: ${run_name}" >&2
        echo "Set RUN_PREFIX to a new value or remove the existing outputs." >&2
        exit 1
    fi

    mkdir -p "${eval_log_dir}"

    echo "============================================================"
    echo "Training format : ${format}"
    echo "Run name        : ${run_name}"
    echo "Train data      : ${train_file}"
    echo "Test data       : ${test_file}"
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

    echo "Evaluating      : ${final_checkpoint}"
    "${PYTHON_BIN}" "${eval_script}" \
        --checkpoint "${final_checkpoint}" \
        --test-file "${test_file}" \
        --output-file "${eval_output}" \
        --device cuda:0 \
        --dtype "${PRECISION}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --num-beams "${NUM_BEAMS}" \
        2>&1 | tee "${eval_log_dir}/eval.log"

    echo "Completed       : ${run_name}"
    echo "Checkpoint      : ${final_checkpoint}"
    echo "Evaluation JSONL: ${eval_output}"
    echo
done

echo "All three training and evaluation runs completed."
