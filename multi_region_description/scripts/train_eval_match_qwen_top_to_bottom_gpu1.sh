#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/work/MichaelYu/florence-caption}"
PYTHON_BIN="${PYTHON_BIN:-/data/work/MichaelYu/miniconda3/envs/florence-caption/bin/python}"
GPU_INDEX="${GPU_INDEX:-1}"
MIN_FREE_MEMORY_MIB="${MIN_FREE_MEMORY_MIB:-60000}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/ugipc_1231_15words_epoch3_full_handoff/checkpoint}"

DATA_ROOT="${PROJECT_ROOT}/multi_region_description/data"
CHECKPOINT_ROOT="${PROJECT_ROOT}/multi_region_description/checkpoints"
LOG_ROOT="${PROJECT_ROOT}/multi_region_description/logs"
EVAL_RESULT_ROOT="${PROJECT_ROOT}/multi_region_description/eval/results"
EVAL_LOG_ROOT="${PROJECT_ROOT}/multi_region_description/eval/logs"
MATCHING_ROOT="${EVAL_RESULT_ROOT}/description_matching_top_to_bottom"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_multi_region.py"
EVAL_SCRIPT="${PROJECT_ROOT}/multi_region_description/eval/scripts/eval_plan_b.py"
MATCHING_SCRIPT="${PROJECT_ROOT}/multi_region_description/eval/scripts/evaluate_description_matching.py"

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
NUM_BEAMS=1

declare -a EXPERIMENTS=("standard" "cleaned")

dataset_for() {
    case "$1" in
        standard) echo "qwen-person-2to5-description-loc-order-center-top-to-bottom" ;;
        cleaned) echo "qwen-person-2to5-cleaned-description-loc-order-center-top-to-bottom" ;;
    esac
}

run_name_for() {
    case "$1" in
        standard) echo "qwen_person_2to5_loc_order_top_to_bottom_gpu1_ep2_bs16_lr8e6" ;;
        cleaned) echo "qwen_person_2to5_cleaned_loc_order_top_to_bottom_gpu1_ep2_bs16_lr8e6" ;;
    esac
}

target_length_for() {
    case "$1" in
        standard) echo "160" ;;
        cleaned) echo "320" ;;
    esac
}

require_file() { [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }; }

require_file "${PYTHON_BIN}"
require_file "${TRAIN_SCRIPT}"
require_file "${EVAL_SCRIPT}"
require_file "${MATCHING_SCRIPT}"
require_dir "${BASE_MODEL}"

for experiment in "${EXPERIMENTS[@]}"; do
    dataset="$(dataset_for "${experiment}")"
    run_name="$(run_name_for "${experiment}")"
    require_file "${DATA_ROOT}/${dataset}/train.jsonl"
    require_file "${DATA_ROOT}/${dataset}/test.jsonl"
    for output in \
        "${CHECKPOINT_ROOT}/${run_name}" \
        "${LOG_ROOT}/${run_name}" \
        "${EVAL_LOG_ROOT}/${run_name}" \
        "${EVAL_RESULT_ROOT}/${run_name}.jsonl" \
        "${MATCHING_ROOT}/${experiment}"; do
        [[ ! -e "${output}" ]] || { echo "Refusing to overwrite: ${output}" >&2; exit 1; }
    done
done

free_memory_mib="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ ! "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < MIN_FREE_MEMORY_MIB )); then
    echo "GPU ${GPU_INDEX} has ${free_memory_mib:-unknown} MiB free; ${MIN_FREE_MEMORY_MIB} MiB required." >&2
    exit 1
fi

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "${EVAL_RESULT_ROOT}" "${EVAL_LOG_ROOT}" "${MATCHING_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export TOKENIZERS_PARALLELISM=false

echo "Base checkpoint : ${BASE_MODEL}"
echo "GPU             : physical ${GPU_INDEX} -> cuda:0 (${free_memory_mib} MiB free)"
echo "Experiments     : ${EXPERIMENTS[*]}"

for experiment in "${EXPERIMENTS[@]}"; do
    dataset="$(dataset_for "${experiment}")"
    run_name="$(run_name_for "${experiment}")"
    target_length="$(target_length_for "${experiment}")"
    train_file="${DATA_ROOT}/${dataset}/train.jsonl"
    test_file="${DATA_ROOT}/${dataset}/test.jsonl"
    final_checkpoint="${CHECKPOINT_ROOT}/${run_name}/final"
    eval_output="${EVAL_RESULT_ROOT}/${run_name}.jsonl"
    eval_log_dir="${EVAL_LOG_ROOT}/${run_name}"
    matching_dir="${MATCHING_ROOT}/${experiment}"
    mkdir -p "${eval_log_dir}" "${matching_dir}"

    echo "============================================================"
    echo "Experiment: ${experiment}; target length: ${target_length}"
    echo "============================================================"

    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
        --data "${train_file}" --model-path "${BASE_MODEL}" \
        --output-root "${CHECKPOINT_ROOT}" --log-root "${LOG_ROOT}" --run-name "${run_name}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LEARNING_RATE}" \
        --warmup-steps "${WARMUP_STEPS}" --max-grad-norm "${MAX_GRAD_NORM}" \
        --precision "${PRECISION}" --num-workers "${NUM_WORKERS}" --log-steps "${LOG_STEPS}" \
        --seed "${SEED}" --prompt-max-length "${PROMPT_MAX_LENGTH}" --target-max-length "${target_length}"

    require_dir "${final_checkpoint}"
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --checkpoint "${final_checkpoint}" --test-file "${test_file}" --output-file "${eval_output}" \
        --device cuda:0 --dtype "${PRECISION}" --max-new-tokens "${target_length}" --num-beams "${NUM_BEAMS}" \
        2>&1 | tee "${eval_log_dir}/eval.log"

    "${PYTHON_BIN}" - "${eval_output}" "${matching_dir}" <<'PY'
import json
import sys
from pathlib import Path
source = Path(sys.argv[1])
output = Path(sys.argv[2])
matched, mismatched = [], []
for line in source.read_text(encoding="utf-8").splitlines():
    if line.strip():
        record = json.loads(line)
        (matched if len(record["gt_pairs"]) == len(record["pred_pairs"]) else mismatched).append(record)
for name, records in (("count_matched_input.jsonl", matched), ("count_mismatched_records.jsonl", mismatched)):
    with (output / name).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"matching input: matched={len(matched)} mismatched={len(mismatched)}")
PY
    "${PYTHON_BIN}" "${MATCHING_SCRIPT}" \
        --input "${matching_dir}/count_matched_input.jsonl" --output-dir "${matching_dir}" \
        2>&1 | tee "${matching_dir}/description_matching.log"

    echo "Completed ${experiment}: ${eval_output}"
done

echo "Both top-to-bottom experiments completed."
