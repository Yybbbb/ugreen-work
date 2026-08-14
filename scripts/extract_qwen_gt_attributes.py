#!/usr/bin/env python3
"""Extract structured person attributes from the **GT captions** of the test set via vLLM.

This mirrors ``extract_qwen_attributes.py`` (same PROMPT, same Qwen36-35b-caption
model, same batching / payload / vLLM parameters used for the Florence prediction
extraction) but reads the caption from each test sample's ``label`` field (the
ground-truth caption) instead of ``prediction``. The goal is a Qwen-derived
attribute view of the GT caption text — independent of the hand-labelled
``attributes`` field — so that Florence predictions and GT can be compared
through the same extractor.

Output is a companion JSONL (one record per test sample) containing the original
test row plus a new ``qwen_attributes`` field (Qwen's extraction from the GT
caption), along with ``qwen_raw`` and ``qwen_error``.

Run:
    python scripts/extract_qwen_gt_attributes.py \
        --input data/prepared/test.jsonl \
        --output data/prepared/test_qwen_attributes.jsonl \
        --base-url http://127.0.0.1:6097/v1 --model Qwen36-35b-caption \
        --batch-size 4 --workers 16 --timeout 180 --retries 2 --max-tokens 2048
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

# Reuse the exact prompt + vLLM helpers from the prediction extractor so the
# Qwen "method" is identical to how Florence predictions were processed.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from extract_qwen_attributes import (  # noqa: E402
    PROMPT,
    endpoint,
    parse_json,
)

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "prepared" / "test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "prepared" / "test_qwen_attributes.jsonl"


def request_batch_gt(
    batch: Sequence[Mapping[str, Any]], *, base_url: str, model: str,
    timeout: float, retries: int, max_tokens: int,
) -> List[Dict[str, Any]]:
    """Same payload as extract_qwen_attributes.request_batch, caption from ``label``."""
    items = [{"id": str(row.get("sample_id") or row.get("row_index")), "caption": row.get("label", "")} for row in batch]
    user = PROMPT + "\n\nItems:\n" + json.dumps(items, ensure_ascii=False)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise person-attribute extraction service. Output JSON only."},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                endpoint(base_url), data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            content = envelope["choices"][0]["message"]["content"]
            parsed = parse_json(content)
            if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                parsed = parsed["results"]
            elif isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                raise ValueError("response is not a JSON array")
            by_id = {str(x.get("id")): x for x in parsed if isinstance(x, dict) and x.get("id") is not None}
            if set(by_id) != {x["id"] for x in items}:
                raise ValueError(f"response ids mismatch: expected {len(items)}, got {len(by_id)}")
            return [{"id": item["id"], "attributes": by_id[item["id"]], "raw": content, "error": None} for item in items]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2.0 * (attempt + 1), 8.0))
    return [{"id": item["id"], "attributes": None, "raw": "", "error": last_error} for item in items]


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_done(path: Path) -> Dict[str, Dict[str, Any]]:
    """Resume support: return {sample_id: record} for already-succeeded rows."""
    done: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(row.get("sample_id"))
                if sid and row.get("qwen_error") is None and isinstance(row.get("qwen_attributes"), dict):
                    done[sid] = row
    return done


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_rows = list(iter_rows(args.input))
    valid_rows = [row for row in source_rows if str(row.get("label") or "").strip()]
    empty_ids = [
        str(row.get("sample_id") or row.get("row_index"))
        for row in source_rows
        if not str(row.get("label") or "").strip()
    ]
    ordered_ids = [str(row.get("sample_id") or row.get("row_index")) for row in valid_rows]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("duplicate sample IDs in GT extraction input")

    fresh = bool(getattr(args, "fresh", False))
    done = {} if fresh else load_done(args.output)
    valid_id_set = set(ordered_ids)
    completed_rows: Dict[str, Dict[str, Any]] = {
        sid: row for sid, row in done.items() if sid in valid_id_set
    }
    pending: List[Dict[str, Any]] = []
    submitted = 0
    completed = 0
    errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures: Dict[concurrent.futures.Future[List[Dict[str, Any]]], Sequence[Mapping[str, Any]]] = {}

        def drain(future, batch_rows):
            nonlocal completed, errors
            sources = {
                str(row.get("sample_id") or row.get("row_index")): row
                for row in batch_rows
            }
            for result in future.result():
                source = sources.get(result["id"])
                if source is None:
                    continue
                record = dict(source)
                record["qwen_attributes"] = result["attributes"]
                record["qwen_raw"] = result["raw"]
                record["qwen_error"] = result["error"]
                completed_rows[result["id"]] = record
                completed += 1
                errors += int(result["error"] is not None)

        for row in valid_rows:
            sid = str(row.get("sample_id") or row.get("row_index"))
            if sid in completed_rows:
                continue
            pending.append(row)
            if len(pending) < args.batch_size:
                continue
            future = pool.submit(request_batch_gt, tuple(pending), base_url=args.base_url, model=args.model,
                                 timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens)
            futures[future] = tuple(pending)
            submitted += len(pending)
            pending = []
            while len(futures) >= args.workers * 2:
                finished = next(iter(concurrent.futures.as_completed(futures)))
                drain(finished, futures.pop(finished))
        if pending:
            future = pool.submit(request_batch_gt, tuple(pending), base_url=args.base_url, model=args.model,
                                 timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens)
            futures[future] = tuple(pending)
            submitted += len(pending)
        for future, batch_rows in list(futures.items()):
            drain(future, batch_rows)

    initial_failed_ids = [
        sid for sid in ordered_ids
        if sid not in completed_rows
        or completed_rows[sid].get("qwen_error") is not None
        or not isinstance(completed_rows[sid].get("qwen_attributes"), dict)
    ]
    rows_by_id = {
        str(row.get("sample_id") or row.get("row_index")): row
        for row in valid_rows
    }
    for sid in initial_failed_ids:
        source = rows_by_id[sid]
        result = request_batch_gt(
            (source,),
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            max_tokens=args.max_tokens,
        )[0]
        record = dict(source)
        record["qwen_attributes"] = result["attributes"]
        record["qwen_raw"] = result["raw"]
        record["qwen_error"] = result["error"]
        completed_rows[sid] = record

    failed_ids = [
        sid for sid in ordered_ids
        if sid not in completed_rows
        or completed_rows[sid].get("qwen_error") is not None
        or not isinstance(completed_rows[sid].get("qwen_attributes"), dict)
    ]
    failures_path = args.output.with_suffix(".failures.jsonl")
    if failed_ids:
        _write_jsonl_atomic(failures_path, [completed_rows[sid] for sid in failed_ids])
        raise RuntimeError(
            f"GT Qwen extraction incomplete: {len(failed_ids)} failed or missing rows; "
            f"first IDs: {failed_ids[:10]}"
        )
    if failures_path.exists():
        failures_path.unlink()

    ordered_output = [completed_rows[sid] for sid in ordered_ids]
    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "caption_source": "label (GT caption)",
        "model": args.model,
        "base_url": args.base_url,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "input_samples": len(source_rows),
        "empty_caption_samples": len(empty_ids),
        "empty_caption_ids": empty_ids,
        "samples": len(ordered_output),
        "submitted": submitted + len(initial_failed_ids),
        "completed": len(ordered_output),
        "initial_errors": len(initial_failed_ids),
        "individual_retries": len(initial_failed_ids),
        "errors": len(failed_ids),
        "resumed_from": len(done),
        "fresh": fresh,
    }
    _write_jsonl_atomic(args.output, ordered_output)
    _write_json_atomic(args.output.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:6097/v1"))
    parser.add_argument("--model", default="Qwen36-35b-caption")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--fresh", action="store_true", help="ignore published output and rebuild it")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        parser.error("batch-size and workers must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
