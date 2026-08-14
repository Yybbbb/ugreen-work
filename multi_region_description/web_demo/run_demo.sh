#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python}"
CHECKPOINT_PLAN_A="${CHECKPOINT_PLAN_A:-/data/work/MichaelYu/florence-caption/multi_region_description/checkpoints/multi_region_no_name_ep2_lr1e5/final}"
CHECKPOINT_PLAN_B="${CHECKPOINT_PLAN_B:-/data/work/MichaelYu/florence-caption/multi_region_description/checkpoints/multi_region_plan_b_ep2_lr1e5/final}"
PORT="${PORT:-7863}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda:1}"
PRECISION="${PRECISION:-fp16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

export FLORENCE_CHECKPOINT_PLAN_A="${CHECKPOINT_PLAN_A}"
export FLORENCE_CHECKPOINT_PLAN_B="${CHECKPOINT_PLAN_B}"
export FLORENCE_MULTI_REGION_PRECISION="${PRECISION}"
export FLORENCE_MULTI_REGION_DEVICE="${DEVICE}"
export FLORENCE_MULTI_REGION_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "checkpoint A : ${CHECKPOINT_PLAN_A}"
echo "checkpoint B : ${CHECKPOINT_PLAN_B}"
echo "device       : ${DEVICE}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "precision    : ${PRECISION}"
echo "max_tokens   : ${MAX_NEW_TOKENS}"
echo "url          : http://localhost:${PORT}"
echo ""

"${PYTHON_BIN}" -m uvicorn server:app --host "${HOST}" --port "${PORT}" --reload false
