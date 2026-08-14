#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data1/work/MichaelYu/florence-attibute"
TORCHRUN="/data1/work/MichaelYu/miniconda3/envs/florence/bin/torchrun"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/rl"

QWEN_RUN="region_category_person_scst_reviewed5981_qwen4b"
LEXICAL_RUN="region_category_person_scst_reviewed5981_lexical"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="3,4,5,6"
export PYTHONUNBUFFERED="1"

if [[ -e "${OUTPUT_ROOT}/${QWEN_RUN}" || -e "${OUTPUT_ROOT}/${LEXICAL_RUN}" ]]; then
  echo "Refusing to overwrite an existing RL run directory." >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] starting ${QWEN_RUN} on physical GPUs ${CUDA_VISIBLE_DEVICES}"
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  scripts/train_person_attribute_rl.py \
  --run-name "${QWEN_RUN}" \
  --reward-matcher qwen \
  --judge-request-concurrency 16 \
  --ddp-timeout-minutes 60 \
  2>&1 | tee "${OUTPUT_ROOT}/${QWEN_RUN}.console.log"

echo "[$(date --iso-8601=seconds)] completed ${QWEN_RUN}; starting ${LEXICAL_RUN}"
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  scripts/train_person_attribute_rl.py \
  --run-name "${LEXICAL_RUN}" \
  --reward-matcher lexical \
  --judge-request-concurrency 16 \
  --ddp-timeout-minutes 60 \
  2>&1 | tee "${OUTPUT_ROOT}/${LEXICAL_RUN}.console.log"

echo "[$(date --iso-8601=seconds)] completed both RL runs"
