#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/work/MichaelYu/florence-caption"
PYTHON_BIN="/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python"
GPU_INDEX=1
MIN_COMPUTE_CAPABILITY_MAJOR=8
MIN_FREE_MEMORY_MIB=60000

BASE_MODEL="${PROJECT_ROOT}/ugipc_1231_15words_epoch3_full_handoff/checkpoint"
DATA_DIR="${PROJECT_ROOT}/multi_region_description/data/qwen-person-2to5-cleaned-description-loc"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/test.jsonl"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_multi_region.py"
EVAL_SCRIPT="${PROJECT_ROOT}/multi_region_description/eval/scripts/eval_plan_b.py"

RUN_NAME="qwen_person_2to5_cleaned_loc_gpu1_ep2_bs16_lr8e6"
CHECKPOINT_ROOT="${PROJECT_ROOT}/multi_region_description/checkpoints"
LOG_ROOT="${PROJECT_ROOT}/multi_region_description/logs"
EVAL_LOG_DIR="${PROJECT_ROOT}/multi_region_description/eval/logs/${RUN_NAME}"
EVAL_OUTPUT="${PROJECT_ROOT}/multi_region_description/eval/results/${RUN_NAME}.jsonl"
FINAL_CHECKPOINT="${CHECKPOINT_ROOT}/${RUN_NAME}/final"

EPOCHS=2
BATCH_SIZE=16
LEARNING_RATE="8e-6"
WARMUP_STEPS=50
MAX_GRAD_NORM=1.0
PRECISION="bf16"
NUM_WORKERS=8
LOG_STEPS=10
SEED=42
PROMPT_MAX_LENGTH=768
TARGET_MAX_LENGTH=320
MAX_NEW_TOKENS=320
NUM_BEAMS=1

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
    CHECK_ONLY=true
    shift
fi
if (( $# != 0 )); then
    echo "Usage: $0 [--check-only]" >&2
    exit 2
fi

for required_file in "${PYTHON_BIN}" "${TRAIN_FILE}" "${TEST_FILE}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "Base checkpoint not found: ${BASE_MODEL}" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required but was not found." >&2
    exit 1
fi

gpu_info="$({
    nvidia-smi \
        --id="${GPU_INDEX}" \
        --query-gpu=name,compute_cap,memory.total,memory.free \
        --format=csv,noheader,nounits
} 2>/dev/null)" || {
    echo "Could not query physical GPU ${GPU_INDEX}." >&2
    exit 1
}

IFS=',' read -r gpu_name compute_capability total_memory_mib free_memory_mib <<<"${gpu_info}"
gpu_name="$(xargs <<<"${gpu_name}")"
compute_capability="$(xargs <<<"${compute_capability}")"
total_memory_mib="$(xargs <<<"${total_memory_mib}")"
free_memory_mib="$(xargs <<<"${free_memory_mib}")"
compute_capability_major="${compute_capability%%.*}"

if [[ ! "${compute_capability_major}" =~ ^[0-9]+$ ]] || (( compute_capability_major < MIN_COMPUTE_CAPABILITY_MAJOR )); then
    echo "GPU ${GPU_INDEX} (${gpu_name}) does not meet the BF16 compute capability requirement." >&2
    echo "Required major capability: ${MIN_COMPUTE_CAPABILITY_MAJOR}; reported: ${compute_capability:-unknown}" >&2
    exit 1
fi

if [[ ! "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < MIN_FREE_MEMORY_MIB )); then
    echo "GPU ${GPU_INDEX} (${gpu_name}) supports BF16 but lacks free memory for this run." >&2
    echo "Required free memory: ${MIN_FREE_MEMORY_MIB} MiB; reported: ${free_memory_mib:-unknown} MiB." >&2
    exit 1
fi

if [[ -e "${CHECKPOINT_ROOT}/${RUN_NAME}" || -e "${LOG_ROOT}/${RUN_NAME}" || -e "${EVAL_LOG_DIR}" || -e "${EVAL_OUTPUT}" ]]; then
    echo "Refusing to overwrite outputs for ${RUN_NAME}." >&2
    exit 1
fi

train_samples="$(wc -l < "${TRAIN_FILE}" | tr -d '[:space:]')"
test_samples="$(wc -l < "${TEST_FILE}" | tr -d '[:space:]')"

echo "Preflight passed."
echo "Run name          : ${RUN_NAME}"
echo "GPU               : physical ${GPU_INDEX} (${gpu_name})"
echo "Compute capability: ${compute_capability}"
echo "GPU memory        : ${free_memory_mib}/${total_memory_mib} MiB free"
echo "Train/test samples: ${train_samples}/${test_samples}"
echo "Epochs/batch/LR   : ${EPOCHS}/${BATCH_SIZE}/${LEARNING_RATE}"
echo "Prompt/target max : ${PROMPT_MAX_LENGTH}/${TARGET_MAX_LENGTH}"

if [[ "${CHECK_ONLY}" == true ]]; then
    echo "Check-only mode: training was not started."
    exit 0
fi

mkdir -p "${EVAL_LOG_DIR}" "$(dirname "${EVAL_OUTPUT}")"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --data "${TRAIN_FILE}" \
    --model-path "${BASE_MODEL}" \
    --output-root "${CHECKPOINT_ROOT}" \
    --log-root "${LOG_ROOT}" \
    --run-name "${RUN_NAME}" \
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

if [[ ! -d "${FINAL_CHECKPOINT}" ]]; then
    echo "Final checkpoint was not created: ${FINAL_CHECKPOINT}" >&2
    exit 1
fi

echo "Training completed. Starting evaluation."

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${FINAL_CHECKPOINT}" \
    --test-file "${TEST_FILE}" \
    --output-file "${EVAL_OUTPUT}" \
    --device cuda:0 \
    --dtype "${PRECISION}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-beams "${NUM_BEAMS}" \
    2>&1 | tee "${EVAL_LOG_DIR}/eval.log"

echo "Training and evaluation completed."
echo "Checkpoint: ${FINAL_CHECKPOINT}"
echo "Results   : ${EVAL_OUTPUT}"
