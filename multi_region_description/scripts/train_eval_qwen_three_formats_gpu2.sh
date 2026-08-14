#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/work/MichaelYu/florence-caption"
RUNNER="${PROJECT_ROOT}/multi_region_description/scripts/train_eval_qwen_three_formats.sh"

GPU_INDEX=2
MIN_FREE_MEMORY_MIB=60000

if [[ ! -x "${RUNNER}" ]]; then
    echo "Training/evaluation runner is not executable: ${RUNNER}" >&2
    exit 1
fi

free_memory_mib="$({
    nvidia-smi \
        --id="${GPU_INDEX}" \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits
} | tr -d '[:space:]')"

if [[ ! "${free_memory_mib}" =~ ^[0-9]+$ ]]; then
    echo "Unable to read free memory for GPU ${GPU_INDEX}." >&2
    exit 1
fi

if (( free_memory_mib < MIN_FREE_MEMORY_MIB )); then
    echo "GPU ${GPU_INDEX} has only ${free_memory_mib} MiB free; at least ${MIN_FREE_MEMORY_MIB} MiB is required." >&2
    exit 1
fi

echo "GPU ${GPU_INDEX} free memory: ${free_memory_mib} MiB"
echo "Starting three sequential training and evaluation runs."

export GPU_ID="${GPU_INDEX}"
export RUN_PREFIX="qwen_person_2to5_gpu2_v1"
export EPOCHS=2
export BATCH_SIZE=16
export LEARNING_RATE=8e-6
export WARMUP_STEPS=50
export MAX_GRAD_NORM=1.0
export PRECISION=bf16
export NUM_WORKERS=8
export LOG_STEPS=10
export SEED=42
export PROMPT_MAX_LENGTH=768
export TARGET_MAX_LENGTH=160
export MAX_NEW_TOKENS=160
export NUM_BEAMS=1

exec "${RUNNER}"
