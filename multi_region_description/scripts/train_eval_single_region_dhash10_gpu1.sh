#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/work/MichaelYu/florence-caption"
TASK_ROOT="${PROJECT_ROOT}/multi_region_description"
PYTHON_BIN="/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python"
GPU_INDEX=1
MIN_FREE_MEMORY_MIB=60000

BASE_MODEL="${PROJECT_ROOT}/ugipc_1231_15words_epoch3_full_handoff/checkpoint"
DATA_DIR="${TASK_ROOT}/data/qwen-person-2to6-cleaned-dhash10-single-region"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/test.jsonl"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_multi_region.py"
EVAL_SCRIPT="${TASK_ROOT}/eval/scripts/eval_single_region_description.py"

RUN_NAME="qwen_single_region_dhash10_grouped_gpu1_ep1_bs24_lr4e6"
CHECKPOINT_ROOT="${TASK_ROOT}/checkpoints"
LOG_ROOT="${TASK_ROOT}/logs"
EVAL_DIR="${TASK_ROOT}/eval/results/${RUN_NAME}"
FINAL_CHECKPOINT="${CHECKPOINT_ROOT}/${RUN_NAME}/final"

EPOCHS=1
BATCH_SIZE=24
LEARNING_RATE="4e-6"
WARMUP_STEPS=35
MAX_GRAD_NORM=1.0
PRECISION="bf16"
NUM_WORKERS=8
LOG_STEPS=10
SEED=42
PROMPT_MAX_LENGTH=640
TARGET_MAX_LENGTH=96
EVAL_BATCH_SIZE=16
MAX_NEW_TOKENS=96
NUM_BEAMS=1

for required in "${PYTHON_BIN}" "${TRAIN_FILE}" "${TEST_FILE}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done
[[ -d "${BASE_MODEL}" ]] || { echo "Missing base model: ${BASE_MODEL}" >&2; exit 1; }

gpu_info="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits)"
IFS=',' read -r gpu_name total_memory_mib free_memory_mib <<<"${gpu_info}"
gpu_name="$(xargs <<<"${gpu_name}")"
total_memory_mib="$(xargs <<<"${total_memory_mib}")"
free_memory_mib="$(xargs <<<"${free_memory_mib}")"
if (( free_memory_mib < MIN_FREE_MEMORY_MIB )); then
    echo "GPU ${GPU_INDEX} has only ${free_memory_mib} MiB free; require ${MIN_FREE_MEMORY_MIB}." >&2
    exit 1
fi

for output in "${CHECKPOINT_ROOT}/${RUN_NAME}" "${LOG_ROOT}/${RUN_NAME}" "${EVAL_DIR}"; do
    [[ ! -e "${output}" ]] || { echo "Refusing to overwrite: ${output}" >&2; exit 1; }
done

echo "Run name       : ${RUN_NAME}"
echo "GPU            : physical ${GPU_INDEX} (${gpu_name}), ${free_memory_mib}/${total_memory_mib} MiB free"
echo "Train/test     : $(wc -l < "${TRAIN_FILE}")/$(wc -l < "${TEST_FILE}")"
echo "Epoch/batch/LR : ${EPOCHS}/${BATCH_SIZE}/${LEARNING_RATE}"
echo "Warmup         : ${WARMUP_STEPS}"
echo "Prompt/target  : ${PROMPT_MAX_LENGTH}/${TARGET_MAX_LENGTH}"

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

[[ -d "${FINAL_CHECKPOINT}" ]] || { echo "Final checkpoint missing: ${FINAL_CHECKPOINT}" >&2; exit 1; }
mkdir -p "${EVAL_DIR}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${FINAL_CHECKPOINT}" \
    --test-file "${TEST_FILE}" \
    --output-file "${EVAL_DIR}/predictions.jsonl" \
    --summary-file "${EVAL_DIR}/summary.json" \
    --device cuda:0 \
    --dtype bf16 \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-beams "${NUM_BEAMS}"

echo "Training and evaluation completed: ${RUN_NAME}"
