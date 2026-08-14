#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/work/MichaelYu/florence-caption"
PYTHON_BIN="/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python"
GPU_INDEX=2
MIN_FREE_MEMORY_MIB=60000

BASE_MODEL="${PROJECT_ROOT}/ugipc_1231_15words_epoch3_full_handoff/checkpoint"
TRAIN_FILE="${PROJECT_ROOT}/multi_region_description/data/qwen-person-2to5-description-loc/train.jsonl"
TEST_FILE="${PROJECT_ROOT}/multi_region_description/data/qwen-person-2to5-description-loc/test.jsonl"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_multi_region.py"
EVAL_SCRIPT="${PROJECT_ROOT}/multi_region_description/eval/scripts/eval_plan_b.py"

RUN_NAME="qwen_person_2to5_loc_gpu2_ep2_bs16_lr8e6"
CHECKPOINT_ROOT="${PROJECT_ROOT}/multi_region_description/checkpoints"
LOG_ROOT="${PROJECT_ROOT}/multi_region_description/logs"
EVAL_LOG_DIR="${PROJECT_ROOT}/multi_region_description/eval/logs/${RUN_NAME}"
EVAL_OUTPUT="${PROJECT_ROOT}/multi_region_description/eval/results/${RUN_NAME}.jsonl"
FINAL_CHECKPOINT="${CHECKPOINT_ROOT}/${RUN_NAME}/final"

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

free_memory_mib="$(
    nvidia-smi \
        --id="${GPU_INDEX}" \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits | tr -d '[:space:]'
)"

if [[ ! "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < MIN_FREE_MEMORY_MIB )); then
    echo "GPU ${GPU_INDEX} does not have the required ${MIN_FREE_MEMORY_MIB} MiB free memory." >&2
    echo "Reported free memory: ${free_memory_mib:-unknown} MiB" >&2
    exit 1
fi

if [[ -e "${CHECKPOINT_ROOT}/${RUN_NAME}" || -e "${LOG_ROOT}/${RUN_NAME}" || -e "${EVAL_LOG_DIR}" || -e "${EVAL_OUTPUT}" ]]; then
    echo "Refusing to overwrite outputs for ${RUN_NAME}." >&2
    exit 1
fi

mkdir -p "${EVAL_LOG_DIR}" "$(dirname "${EVAL_OUTPUT}")"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export TOKENIZERS_PARALLELISM=false

echo "Training ${RUN_NAME} on physical GPU ${GPU_INDEX} with ${free_memory_mib} MiB free."

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --data "${TRAIN_FILE}" \
    --model-path "${BASE_MODEL}" \
    --output-root "${CHECKPOINT_ROOT}" \
    --log-root "${LOG_ROOT}" \
    --run-name "${RUN_NAME}" \
    --epochs 2 \
    --batch-size 16 \
    --lr 8e-6 \
    --warmup-steps 50 \
    --max-grad-norm 1.0 \
    --precision bf16 \
    --num-workers 8 \
    --log-steps 10 \
    --seed 42 \
    --prompt-max-length 768 \
    --target-max-length 160

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
    --dtype bf16 \
    --max-new-tokens 160 \
    --num-beams 1 \
    2>&1 | tee "${EVAL_LOG_DIR}/eval.log"

echo "Training and evaluation completed."
echo "Checkpoint: ${FINAL_CHECKPOINT}"
echo "Results   : ${EVAL_OUTPUT}"
