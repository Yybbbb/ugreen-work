#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/data1/work/MichaelYu/florence-attibute"
PYTHON="/data1/work/MichaelYu/miniconda3/envs/florence/bin/python"
TORCHRUN="/data1/work/MichaelYu/miniconda3/envs/florence/bin/torchrun"
TEST_DATA="$PROJECT_ROOT/data/prepared/test.jsonl"
OUTPUT_ROOT="$PROJECT_ROOT/artifacts/rl/evaluation"
RUNNER_LOG="$OUTPUT_ROOT/runner.log"

V4B_CHECKPOINT="$PROJECT_ROOT/artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final"
QWEN_CHECKPOINT="$PROJECT_ROOT/artifacts/rl/region_category_person_scst_reviewed5981_qwen4b/final"
LEXICAL_CHECKPOINT="$PROJECT_ROOT/artifacts/rl/region_category_person_scst_reviewed5981_lexical/final"

V4B_PREDICTIONS="$PROJECT_ROOT/artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/test_inference_final/predictions.jsonl"
QWEN_INFERENCE="$OUTPUT_ROOT/qwen_rl/inference"
LEXICAL_INFERENCE="$OUTPUT_ROOT/lexical_rl/inference"

EXPECTED_ROWS=4328

# Reuse complete, row-accurate Florence predictions when they already exist. The
# evaluate CLI deep-validates checkpoint + frozen generation, so a light row-count
# check is sufficient here; this keeps reruns from re-inferring.
reuse_predictions() {
  local pred="$1"
  [[ -s "$pred" ]] && [[ "$(wc -l < "$pred" | tr -d ' ')" -eq "$EXPECTED_ROWS" ]]
}

mkdir -p "$OUTPUT_ROOT" "$QWEN_INFERENCE" "$LEXICAL_INFERENCE"
echo $$ > "$OUTPUT_ROOT/runner.pid"
exec > >(tee -a "$RUNNER_LOG") 2>&1

on_exit() {
  status=$?
  rm -f "$OUTPUT_ROOT/runner.pid"
  if [[ $status -ne 0 ]]; then
    date -Is > "$OUTPUT_ROOT/evaluation.failed"
    echo "[$(date -Is)] evaluation failed with status $status"
  fi
}
trap on_exit EXIT

rm -f "$OUTPUT_ROOT/evaluation.complete" "$OUTPUT_ROOT/evaluation.failed"
echo "[$(date -Is)] waiting for lexical RL final checkpoint"
while [[ ! -s "${LEXICAL_CHECKPOINT}/model.safetensors" ]]; do
  if ! pgrep -f 'train_person_attribute_rl.py.*region_category_person_scst_reviewed5981_lexical' >/dev/null; then
    echo "lexical training exited before final checkpoint was written" >&2
    exit 1
  fi
  sleep 60
done

while pgrep -f 'train_person_attribute_rl.py.*region_category_person_scst_reviewed5981_lexical' >/dev/null; do
  echo "[$(date -Is)] lexical final exists; waiting for training processes to release GPU 3-6"
  sleep 15
done

for required in \
  "$V4B_CHECKPOINT/model.safetensors" \
  "$QWEN_CHECKPOINT/model.safetensors" \
  "$LEXICAL_CHECKPOINT/model.safetensors" \
  "$V4B_PREDICTIONS" \
  "$TEST_DATA"; do
  if [[ ! -s "$required" ]]; then
    echo "required evaluation input is missing or empty: $required" >&2
    exit 1
  fi
done

cd "$PROJECT_ROOT"
if reuse_predictions "$QWEN_INFERENCE/predictions.jsonl"; then
  echo "[$(date -Is)] reusing existing qwen_rl predictions ($EXPECTED_ROWS rows)"
else
  echo "[$(date -Is)] starting Qwen RL test inference on GPU 3-6"
  CUDA_VISIBLE_DEVICES=3,4,5,6 "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/infer_person_attribute.py \
    --checkpoint "$QWEN_CHECKPOINT" \
    --test-data "$TEST_DATA" \
    --output-dir "$QWEN_INFERENCE" \
    --batch-size 4 \
    --max-new-tokens 64
fi

if reuse_predictions "$LEXICAL_INFERENCE/predictions.jsonl"; then
  echo "[$(date -Is)] reusing existing lexical_rl predictions ($EXPECTED_ROWS rows)"
else
  echo "[$(date -Is)] starting lexical RL test inference on GPU 3-6"
  CUDA_VISIBLE_DEVICES=3,4,5,6 "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/infer_person_attribute.py \
    --checkpoint "$LEXICAL_CHECKPOINT" \
    --test-data "$TEST_DATA" \
    --output-dir "$LEXICAL_INFERENCE" \
    --batch-size 4 \
    --max-new-tokens 64
fi

echo "[$(date -Is)] starting unified extraction, semantic judging, and report generation"
"$PYTHON" scripts/evaluate_person_attribute_rl.py \
  --candidate "v4b_sft|$V4B_CHECKPOINT|$V4B_PREDICTIONS" \
  --candidate "qwen_rl|$QWEN_CHECKPOINT|$QWEN_INFERENCE/predictions.jsonl" \
  --candidate "lexical_rl|$LEXICAL_CHECKPOINT|$LEXICAL_INFERENCE/predictions.jsonl" \
  --test-data "$TEST_DATA" \
  --output-dir "$OUTPUT_ROOT" \
  --expected-samples 4328 \
  --extractor-base-url http://127.0.0.1:6097/v1 \
  --extractor-model Qwen3.6-27B-FP8 \
  --extractor-batch-size 4 \
  --extractor-workers 32 \
  --extractor-max-tokens 2048 \
  --judge-base-url http://127.0.0.1:6098/v1 \
  --judge-model Qwen3.5-4B \
  --judge-workers 64 \
  --bootstrap-replicates 2000 \
  --seed 20260720

date -Is > "$OUTPUT_ROOT/evaluation.complete"
trap - EXIT
echo "[$(date -Is)] evaluation complete"
