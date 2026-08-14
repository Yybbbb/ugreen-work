#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python}"
CHECKPOINT="${CHECKPOINT:-/data/work/MichaelYu/florence-caption/multi_region_description/checkpoints/qwen_single_region_dhash10_grouped_gpu1_ep1_bs24_lr4e6/final}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8512}"
GPU_INDEX="${GPU_INDEX:-1}"
PRECISION="${PRECISION:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
NUM_BEAMS="${NUM_BEAMS:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export TOKENIZERS_PARALLELISM=false
export FLORENCE_SINGLE_REGION_CHECKPOINT="${CHECKPOINT}"
export FLORENCE_SINGLE_REGION_DEVICE="cuda:0"
export FLORENCE_SINGLE_REGION_PRECISION="${PRECISION}"
export FLORENCE_SINGLE_REGION_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
export FLORENCE_SINGLE_REGION_NUM_BEAMS="${NUM_BEAMS}"

echo "checkpoint : ${CHECKPOINT}"
echo "GPU        : physical ${GPU_INDEX} -> cuda:0"
echo "precision  : ${PRECISION}"
echo "URL        : http://${HOST}:${PORT}"

exec "${PYTHON_BIN}" -m uvicorn server:app --host "${HOST}" --port "${PORT}"
