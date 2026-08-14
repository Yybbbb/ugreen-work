#!/usr/bin/env python3
"""Resumable unified test evaluation for V4B SFT and two RL checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from evaluate_qwen_attribute_extraction import (
    SCALAR_FIELDS,
    normalize_extra,
    normalize_value,
    normalize_value_set,
)
from rl_clients import extract_attributes, make_judge_fn, preflight_service
from rl_test_metrics import build_candidate, compare_candidates, render_markdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAMES = ("v4b_sft", "qwen_rl", "lexical_rl")
FROZEN_GENERATION = {"do_sample": False, "num_beams": 1, "max_new_tokens": 64}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def load_gt_attributes(path: Optional[Path]) -> Dict[str, Mapping[str, Any]]:
    """Load a {sample_id: attributes} GT map from a jsonl (e.g. qwen_attributes).

    Each row must carry ``sample_id`` and ``qwen_attributes`` (or ``attributes``).
    Returns an empty dict when ``path`` is None, leaving the caller to fall back
    to the raw ``test.jsonl`` attributes (the legacy, list-valued source).
    """
    if path is None:
        return {}
    mapping: Dict[str, Mapping[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        attrs = row.get("qwen_attributes")
        if not isinstance(attrs, Mapping):
            attrs = row.get("attributes")
        if isinstance(attrs, Mapping):
            mapping[sample_id] = attrs
    return mapping


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_frozen_config(path: Path, config: Mapping[str, Any]) -> None:
    """Create the resume contract once and reject any later configuration drift."""
    normalized = json.loads(json.dumps(config, ensure_ascii=False, sort_keys=True))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(
                f"evaluation resume configuration mismatch at {path}; "
                "use a new output directory for a changed protocol"
            )
        return
    _write_json_atomic(path, normalized)


def validate_predictions(
    test_path: Path,
    predictions_path: Path,
    checkpoint: Path,
    *,
    expected_samples: int,
    max_samples: int = 0,
    gt_attributes: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    tests = list(iter_jsonl(test_path))
    predictions = list(iter_jsonl(predictions_path))
    if max_samples:
        tests = tests[:max_samples]
        predictions = predictions[:max_samples]
    if len(tests) != expected_samples or len(predictions) != expected_samples:
        raise ValueError(
            f"sample count mismatch: test={len(tests)} predictions={len(predictions)} "
            f"expected={expected_samples}"
        )
    test_ids = [str(row.get("sample_id") or "") for row in tests]
    prediction_ids = [str(row.get("sample_id") or "") for row in predictions]
    if not all(test_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("test sample IDs must be non-empty and unique")
    if prediction_ids != test_ids:
        raise ValueError("prediction sample IDs must exactly match test order")
    expected_checkpoint = str(Path(checkpoint).resolve())
    rows = []
    for test, prediction in zip(tests, predictions):
        actual_checkpoint = str(Path(str(prediction.get("checkpoint") or "")).resolve())
        if actual_checkpoint != expected_checkpoint:
            raise ValueError(
                f"checkpoint mismatch for {test['sample_id']}: "
                f"{actual_checkpoint!r} != {expected_checkpoint!r}"
            )
        if prediction.get("generation") != FROZEN_GENERATION:
            raise ValueError(f"generation mismatch for {test['sample_id']}")
        combined = dict(prediction)
        sample_id = str(test.get("sample_id") or "")
        gt_attrs = None
        if gt_attributes is not None:
            candidate = gt_attributes.get(sample_id)
            if isinstance(candidate, Mapping):
                gt_attrs = candidate
        if gt_attrs is None:
            gt_attrs = test.get("attributes") if isinstance(test.get("attributes"), Mapping) else {}
        combined["ground_truth"] = gt_attrs
        combined["session"] = str(test.get("session") or prediction.get("session") or "")
        combined["scene"] = str(test.get("scene") or prediction.get("scene") or "")
        rows.append(combined)
    return rows


def request_extraction_batch(
    batch: Sequence[Mapping[str, Any]],
    *,
    base_url: str,
    model: str,
    timeout: float,
    retries: int,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    attributes = extract_attributes(
        [str(row.get("prediction") or "") for row in batch],
        base_url=base_url,
        model=model,
        timeout=timeout,
        retries=retries,
        max_tokens=max_tokens,
    )
    results = []
    for row, extracted in zip(batch, attributes):
        sample_id = str(row["sample_id"])
        results.append({
            "id": sample_id,
            "attributes": extracted,
            "raw": "",
            "error": None if isinstance(extracted, dict) else "missing extractor result",
        })
    return results


def _successful_journal_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    successful: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return successful
    for row in iter_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if sample_id and row.get("error") is None and isinstance(row.get("attributes"), Mapping):
            successful[sample_id] = row
    return successful


def _extraction_record(source: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(source["sample_id"]),
        "sample_id": str(source["sample_id"]),
        "scene": source.get("scene"),
        "session": source.get("session"),
        "prediction": source.get("prediction"),
        "ground_truth": source.get("ground_truth", {}),
        "attributes": result.get("attributes"),
        "raw": result.get("raw", ""),
        "error": result.get("error"),
    }


def extract_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    journal_path: Path,
    final_path: Path,
    *,
    request_fn: Callable[..., List[Dict[str, Any]]] = request_extraction_batch,
    base_url: str,
    model: str,
    batch_size: int,
    workers: int,
    timeout: float,
    retries: int,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    source_by_id = {str(row["sample_id"]): row for row in rows}
    if len(source_by_id) != len(rows):
        raise ValueError("candidate rows contain duplicate sample IDs")
    completed = {
        sample_id: _extraction_record(source_by_id[sample_id], record)
        for sample_id, record in _successful_journal_rows(journal_path).items()
        if sample_id in source_by_id
    }
    pending = [row for row in rows if str(row["sample_id"]) not in completed]
    batches = [pending[index:index + batch_size] for index in range(0, len(pending), batch_size)]

    def submit(batch):
        return request_fn(
            batch,
            base_url=base_url,
            model=model,
            timeout=timeout,
            retries=retries,
            max_tokens=max_tokens,
        )

    with journal_path.open("a", encoding="utf-8") as journal:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(submit, batch): batch for batch in batches}
            for future in concurrent.futures.as_completed(futures):
                batch = futures[future]
                batch_by_id = {str(row["sample_id"]): row for row in batch}
                for result in future.result():
                    sample_id = str(result.get("id") or "")
                    if sample_id not in batch_by_id:
                        continue
                    record = _extraction_record(batch_by_id[sample_id], result)
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if record["error"] is None and isinstance(record["attributes"], Mapping):
                        completed[sample_id] = record
                journal.flush()
                os.fsync(journal.fileno())

        missing = [row for row in rows if str(row["sample_id"]) not in completed]
        for source in missing:
            result_list = submit([source])
            result = result_list[0] if result_list else {
                "id": source["sample_id"], "attributes": None, "error": "empty extractor response"
            }
            record = _extraction_record(source, result)
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            if record["error"] is None and isinstance(record["attributes"], Mapping):
                completed[str(source["sample_id"])] = record

    unresolved = [str(row["sample_id"]) for row in rows if str(row["sample_id"]) not in completed]
    if unresolved:
        raise RuntimeError(f"unresolved extraction failures: {len(unresolved)}; first={unresolved[:10]}")
    ordered = [completed[str(row["sample_id"])] for row in rows]
    _write_jsonl_atomic(final_path, ordered)
    return ordered


def stream_extract_and_judge(
    candidates_in: Sequence[Tuple[str, Path, Sequence[Mapping[str, Any]]]],
    output_dir: Path,
    *,
    request_fn: Callable[..., List[Dict[str, Any]]] = request_extraction_batch,
    base_url: str,
    model: str,
    batch_size: int,
    workers: int,
    timeout: float,
    retries: int,
    max_tokens: int,
    judge: "PersistentJudge",
    progress_every: int = 25,
) -> List[Tuple[str, Path, List[Mapping[str, Any]]]]:
    """Batch-pipelined extraction and judging across all candidates in one stream.

    Batches from every candidate flow through one extraction pool (the GPU 1
    service). As soon as a batch is extracted, its similarity triples are
    prefetched to the judge (the GPU 2 service), so extraction and judging
    overlap instead of running as separate all-then-all stages. Per-candidate
    extraction journals and the persistent judge cache make the stream resumable:
    completed sample IDs are skipped on restart. Returns one ordered extracted-row
    list per candidate, parallel to the input.
    """
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    states: List[Dict[str, Any]] = []
    for name, checkpoint, rows in candidates_in:
        candidate_dir = output_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        journal_path = candidate_dir / "extraction.journal.jsonl"
        final_path = candidate_dir / "extractions.jsonl"
        source_by_id = {str(row["sample_id"]): row for row in rows}
        if len(source_by_id) != len(rows):
            raise ValueError(f"candidate {name} contains duplicate sample IDs")
        completed = {
            sample_id: _extraction_record(source_by_id[sample_id], record)
            for sample_id, record in _successful_journal_rows(journal_path).items()
            if sample_id in source_by_id
        }
        pending = [row for row in rows if str(row["sample_id"]) not in completed]
        batches = [pending[index:index + batch_size] for index in range(0, len(pending), batch_size)]
        states.append({
            "name": name,
            "checkpoint": checkpoint,
            "rows": rows,
            "journal_path": journal_path,
            "final_path": final_path,
            "completed": completed,
            "batches": batches,
            "handle": None,
        })

    work_units: List[Tuple[int, Sequence[Mapping[str, Any]]]] = [
        (state_index, batch)
        for state_index, state in enumerate(states)
        for batch in state["batches"]
    ]
    total_batches = len(work_units)
    for state in states:
        state["handle"] = state["journal_path"].open("a", encoding="utf-8")

    completed_batches = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    request_fn, batch,
                    base_url=base_url, model=model, timeout=timeout,
                    retries=retries, max_tokens=max_tokens,
                ): (state_index, batch)
                for state_index, batch in work_units
            }
            for future in concurrent.futures.as_completed(futures):
                state_index, batch = futures[future]
                state = states[state_index]
                batch_by_id = {str(row["sample_id"]): row for row in batch}
                completed_records: List[Mapping[str, Any]] = []
                for result in future.result():
                    sample_id = str(result.get("id") or "")
                    if sample_id not in batch_by_id:
                        continue
                    record = _extraction_record(batch_by_id[sample_id], result)
                    state["handle"].write(json.dumps(record, ensure_ascii=False) + "\n")
                    if record["error"] is None and isinstance(record["attributes"], Mapping):
                        state["completed"][sample_id] = record
                        completed_records.append(record)
                state["handle"].flush()
                os.fsync(state["handle"].fileno())
                # Judge this batch's triples now, overlapping GPU 2 with ongoing extraction.
                if completed_records:
                    judge.prefetch(similarity_requests(completed_records))
                completed_batches += 1
                if progress_every and completed_batches % progress_every == 0:
                    logging.info(
                        "stream extract+judge: %d/%d batches across %d candidates",
                        completed_batches, total_batches, len(states),
                    )

        # Retry any still-unresolved samples one at a time, mirroring extract_candidate_rows.
        for state in states:
            missing = [row for row in state["rows"] if str(row["sample_id"]) not in state["completed"]]
            for source in missing:
                result_list = request_fn(
                    [source], base_url=base_url, model=model, timeout=timeout,
                    retries=retries, max_tokens=max_tokens,
                )
                result = result_list[0] if result_list else {
                    "id": source["sample_id"], "attributes": None, "error": "empty extractor response",
                }
                record = _extraction_record(source, result)
                state["handle"].write(json.dumps(record, ensure_ascii=False) + "\n")
                state["handle"].flush()
                os.fsync(state["handle"].fileno())
                if record["error"] is None and isinstance(record["attributes"], Mapping):
                    state["completed"][str(source["sample_id"])] = record
                    judge.prefetch(similarity_requests([record]))
    finally:
        for state in states:
            if state["handle"] is not None:
                state["handle"].close()

    prepared: List[Tuple[str, Path, List[Mapping[str, Any]]]] = []
    for state in states:
        unresolved = [
            str(row["sample_id"]) for row in state["rows"]
            if str(row["sample_id"]) not in state["completed"]
        ]
        if unresolved:
            raise RuntimeError(
                f"unresolved extraction failures for {state['name']}: {len(unresolved)}; first={unresolved[:10]}"
            )
        ordered = [state["completed"][str(row["sample_id"])] for row in state["rows"]]
        _write_jsonl_atomic(state["final_path"], ordered)
        prepared.append((state["name"], state["checkpoint"], ordered))
    return prepared


def _cache_key(field: str, gt_value: str, pred_value: str) -> str:
    return "\x1f".join((field, gt_value, pred_value))


class PersistentJudge:
    def __init__(self, path: Path, network_judge: Callable[[str, str, str], float]):
        self.path = Path(path)
        self.network_judge = network_judge
        self.cache: Dict[str, float] = {}
        if self.path.is_file():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("judge cache must be a JSON object")
            self.cache = {str(key): float(value) for key, value in loaded.items()}

    def prefetch(self, requests: Iterable[Tuple[str, str, str]]) -> None:
        missing = []
        seen = set()
        for field, gt_value, pred_value in requests:
            item = (str(field), str(gt_value), str(pred_value))
            key = _cache_key(*item)
            if key not in self.cache and key not in seen:
                seen.add(key)
                missing.append(item)
        if not missing:
            return
        network_prefetch = getattr(self.network_judge, "prefetch", None)
        if network_prefetch is not None:
            network_prefetch(missing)
        for item in missing:
            self.cache[_cache_key(*item)] = float(self.network_judge(*item))
        _write_json_atomic(self.path, self.cache)

    def __call__(self, field: str, gt_value: str, pred_value: str) -> float:
        key = _cache_key(field, gt_value, pred_value)
        if key not in self.cache:
            self.prefetch([(field, gt_value, pred_value)])
        return self.cache[key]


def _nested(attributes: Mapping[str, Any], path: str) -> Any:
    value: Any = attributes
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def similarity_requests(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str, str]]:
    requests = []
    for row in rows:
        gt = row.get("ground_truth") if isinstance(row.get("ground_truth"), Mapping) else {}
        pred = row.get("attributes") if isinstance(row.get("attributes"), Mapping) else {}
        for field in SCALAR_FIELDS:
            pred_value = normalize_value(field, _nested(pred, field))
            for gt_value in normalize_value_set(field, _nested(gt, field)):
                if pred_value is not None:
                    requests.append((field, gt_value, pred_value))
        for gt_value in normalize_extra(gt.get("extra")):
            for pred_value in normalize_extra(pred.get("extra")):
                requests.append(("extra", gt_value, pred_value))
    return requests


def publish_reports(
    candidates: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 20260720,
) -> Dict[str, Any]:
    names = [str(candidate["name"]) for candidate in candidates]
    if len(candidates) != 3 or set(names) != set(EXPECTED_NAMES):
        raise ValueError(f"reports require exactly these candidates: {EXPECTED_NAMES}")
    ordered = [next(candidate for candidate in candidates if candidate["name"] == name) for name in EXPECTED_NAMES]
    comparison = compare_candidates(
        ordered,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "person_caption_metrics.json", comparison)
    _write_text_atomic(output_dir / "person_caption_comparison.md", render_markdown(comparison))
    return comparison


def parse_candidate(value: str) -> Tuple[str, Path, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("candidate must be NAME|CHECKPOINT|PREDICTIONS")
    return parts[0], Path(parts[1]), Path(parts[2])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    test_hash = sha256_file(args.test_data)
    frozen_config = {
        "test_data": str(args.test_data.resolve()),
        "test_sha256": test_hash,
        "gt_attributes": str(args.gt_attributes.resolve()) if args.gt_attributes else None,
        "gt_attributes_sha256": sha256_file(args.gt_attributes) if args.gt_attributes else None,
        "expected_samples": args.expected_samples,
        "max_samples": args.max_samples,
        "generation": FROZEN_GENERATION,
        "candidates": [
            {"name": name, "checkpoint": str(checkpoint.resolve()), "predictions": str(predictions.resolve())}
            for name, checkpoint, predictions in args.candidate
        ],
        "extractor": {
            "base_url": args.extractor_base_url,
            "model": args.extractor_model,
            "batch_size": args.extractor_batch_size,
            "workers": args.extractor_workers,
            "max_tokens": args.extractor_max_tokens,
        },
        "judge": {
            "base_url": args.judge_base_url,
            "model": args.judge_model,
            "workers": args.judge_workers,
        },
        "timeout": args.timeout,
        "retries": args.retries,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
    }
    ensure_frozen_config(args.output_dir / "run_config.json", frozen_config)
    preflight_service(args.extractor_base_url, args.extractor_model)
    preflight_service(args.judge_base_url, args.judge_model)
    network_judge = make_judge_fn(
        args.judge_base_url,
        args.judge_model,
        timeout=args.timeout,
        retries=args.retries,
        concurrency=args.judge_workers,
    )
    judge = PersistentJudge(args.output_dir / "judge_cache.json", network_judge)
    gt_attributes = load_gt_attributes(args.gt_attributes)
    inputs: List[Tuple[str, Path, List[Mapping[str, Any]]]] = []
    effective_samples = args.max_samples or args.expected_samples
    for name, checkpoint, predictions in args.candidate:
        if name not in EXPECTED_NAMES:
            raise ValueError(f"unknown candidate name: {name}")
        rows = validate_predictions(
            args.test_data,
            predictions,
            checkpoint,
            expected_samples=effective_samples,
            max_samples=args.max_samples,
            gt_attributes=gt_attributes or None,
        )
        inputs.append((name, checkpoint, rows))
    # Batch-pipelined stream: every candidate's batches flow through one
    # extraction pool, and each completed batch is judged immediately so the
    # GPU 1 extractor and GPU 2 judge overlap (no all-then-all staging).
    prepared = stream_extract_and_judge(
        inputs,
        args.output_dir,
        request_fn=request_extraction_batch,
        base_url=args.extractor_base_url,
        model=args.extractor_model,
        batch_size=args.extractor_batch_size,
        workers=args.extractor_workers,
        timeout=args.timeout,
        retries=args.retries,
        max_tokens=args.extractor_max_tokens,
        judge=judge,
    )

    candidates = []
    for name, checkpoint, rows in prepared:
        candidate = build_candidate(name, str(checkpoint.resolve()), rows, judge)
        _write_json_atomic(args.output_dir / name / "candidate_scores.json", candidate)
        candidates.append(candidate)
    result = publish_reports(
        candidates,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    _write_json_atomic(args.output_dir / "evaluation_manifest.json", {
        "test_data": str(args.test_data.resolve()),
        "test_sha256": test_hash,
        "samples": effective_samples,
        "max_samples": args.max_samples,
        "extractor": {"base_url": args.extractor_base_url, "model": args.extractor_model},
        "judge": {"base_url": args.judge_base_url, "model": args.judge_model},
        "generation": FROZEN_GENERATION,
        "candidates": [
            {"name": name, "checkpoint": str(checkpoint.resolve())}
            for name, checkpoint, unused_rows in prepared
        ],
        "status": "complete",
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--test-data", type=Path, default=PROJECT_ROOT / "data/prepared/test.jsonl")
    parser.add_argument(
        "--gt-attributes",
        type=Path,
        default=PROJECT_ROOT / "data/prepared/test_qwen_attributes.jsonl",
        help="jsonl of GT attributes keyed by sample_id (qwen_attributes field). "
             "Defaults to the reviewed-caption extraction so list-valued GT in "
             "test.jsonl is not mis-scored as fabrication. Set to test.jsonl for "
             "the legacy raw-attributes GT.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/rl/evaluation")
    parser.add_argument("--expected-samples", type=int, default=4328)
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate only the first N rows for smoke tests")
    parser.add_argument("--extractor-base-url", default="http://127.0.0.1:6097/v1")
    parser.add_argument("--extractor-model", default="Qwen3.6-27B-FP8")
    parser.add_argument("--extractor-batch-size", type=int, default=8)
    parser.add_argument("--extractor-workers", type=int, default=8)
    parser.add_argument("--extractor-max-tokens", type=int, default=1024)
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:6098/v1")
    parser.add_argument("--judge-model", default="Qwen3.5-4B")
    parser.add_argument("--judge-workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if len(args.candidate) != 3:
        parser.error("provide exactly three --candidate values")
    if args.max_samples < 0 or (args.max_samples and args.max_samples > args.expected_samples):
        parser.error("max-samples must be zero or in [1, expected-samples]")
    for value in (
        args.expected_samples,
        args.extractor_batch_size,
        args.extractor_workers,
        args.extractor_max_tokens,
        args.judge_workers,
        args.bootstrap_replicates,
    ):
        if value <= 0:
            parser.error("sample, batch, worker, token, and bootstrap values must be positive")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(parse_args())
    print(json.dumps(result["recommendation"], ensure_ascii=False, indent=2), flush=True)
