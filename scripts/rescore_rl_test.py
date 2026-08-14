#!/usr/bin/env python3
"""Re-score the frozen RL test with qwen_attributes GT + list-aware scorer.

The frozen ``artifacts/rl/evaluation`` used ``test.jsonl:attributes`` as GT,
whose list-valued fields (``['yellow']``) the scorer collapsed to ``None`` and
therefore mis-counted as fabrication. This script corrects the metrics without
re-running any GPU inference: it reuses each candidate's existing Florence
predictions and their Qwen extractions (unchanged) and only swaps the GT to
``test_qwen_attributes.jsonl`` (clean strings extracted from the reviewed GT
caption) and re-scores with the now list-aware ``score_attribute_pair``.

Lexical route is pure Python. The semantic route reuses ``judge_cache.json``
and queries the Qwen3.5-4B judge (port 6098) only for new (field, gt, pred)
triples introduced by the GT change.

Run:
    python scripts/rescore_rl_test.py
"""

from __future__ import annotations

import argparse
import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping

from rl_clients import make_judge_fn, preflight_service
from rl_test_metrics import build_candidate, compare_candidates, render_markdown
from evaluate_person_attribute_rl import (
    EXPECTED_NAMES,
    PersistentJudge,
    load_gt_attributes,
    similarity_requests,
)
from person_sft_data import sha256_file, write_json_atomic, write_jsonl_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("rescore_rl_test")


def _load_extractions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _checkpoint_from_scores(path: Path) -> str:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = data.get("checkpoint")
        if isinstance(checkpoint, str) and checkpoint:
            return checkpoint
    return ""


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_attributes = load_gt_attributes(args.gt_attributes)
    if not gt_attributes:
        raise ValueError(f"no GT attributes loaded from {args.gt_attributes}")
    LOGGER.info("loaded %d GT attribute rows from %s", len(gt_attributes), args.gt_attributes)

    preflight_service(args.judge_base_url, args.judge_model)
    network_judge = make_judge_fn(
        args.judge_base_url,
        args.judge_model,
        timeout=args.timeout,
        retries=args.retries,
        concurrency=args.judge_workers,
    )
    judge = PersistentJudge(output_dir / "judge_cache.json", network_judge)

    candidates = []
    missing_gt = 0
    for name in EXPECTED_NAMES:
        extractions_path = input_dir / name / "extractions.jsonl"
        if not extractions_path.is_file():
            raise FileNotFoundError(f"{extractions_path} not found")
        rows = _load_extractions(extractions_path)
        corrected: List[Dict[str, Any]] = []
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            gt = gt_attributes.get(sample_id)
            if not isinstance(gt, Mapping):
                missing_gt += 1
                gt = row.get("ground_truth") if isinstance(row.get("ground_truth"), Mapping) else {}
            new_row = dict(row)
            new_row["ground_truth"] = gt
            corrected.append(new_row)
        checkpoint = _checkpoint_from_scores(input_dir / name / "candidate_scores.json")
        LOGGER.info("prefetching judge triples for %s (%d rows)", name, len(corrected))
        judge.prefetch(similarity_requests(corrected))
        candidate = build_candidate(name, checkpoint, corrected, judge)
        write_json_atomic(output_dir / name / "candidate_scores.json", candidate)
        write_jsonl_atomic(output_dir / name / "extractions.jsonl", corrected)
        candidates.append(candidate)
        LOGGER.info(
            "%s: lexical F1 %.4f fab %.4f | semantic F1 %.4f",
            name,
            candidate["lexical"]["metrics"]["micro"]["f1"],
            candidate["lexical"]["metrics"]["fabrication_ratio"],
            candidate["semantic"]["metrics"]["micro"]["f1"],
        )
    if missing_gt:
        LOGGER.warning("%d rows had no qwen_attributes GT; fell back to raw attributes", missing_gt)

    comparison = compare_candidates(
        candidates,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    write_json_atomic(output_dir / "person_caption_metrics.json", comparison)
    (output_dir / "person_caption_comparison.md").write_text(
        render_markdown(comparison), encoding="utf-8"
    )

    gt_hash = sha256_file(args.gt_attributes) if args.gt_attributes else None
    run_config = {
        "rescore": True,
        "purpose": "qwen_attributes GT + list-aware scorer; corrections of the frozen evaluation",
        "input_dir": str(input_dir.resolve()),
        "gt_attributes": str(args.gt_attributes.resolve()),
        "gt_attributes_sha256": gt_hash,
        "judge": {
            "base_url": args.judge_base_url,
            "model": args.judge_model,
            "workers": args.judge_workers,
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "candidates": [
            {"name": c["name"], "checkpoint": c["checkpoint"]} for c in candidates
        ],
    }
    write_json_atomic(output_dir / "rescore_config.json", run_config)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "artifacts/rl/evaluation_frozen_20260812")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/rl/evaluation")
    parser.add_argument(
        "--gt-attributes",
        type=Path,
        default=PROJECT_ROOT / "data/prepared/test_qwen_attributes.jsonl",
    )
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:6098/v1")
    parser.add_argument("--judge-model", default="Qwen3.5-4B")
    parser.add_argument("--judge-workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(parse_args())
    print(json.dumps(result["recommendation"], ensure_ascii=False, indent=2), flush=True)
