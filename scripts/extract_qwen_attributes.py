#!/usr/bin/env python3
"""Extract structured person attributes from Florence captions via vLLM."""

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "artifacts/sft/region_category_person_sft_30k_replay1k_b64/test_inference_final_v2/predictions.jsonl"
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "qwen_attribute_extraction.jsonl"
PROMPT = (
    "Extract only attributes explicitly stated in each person caption. "
    "Return a JSON object with one key named results containing one object per item, "
    "preserving each id. Every result must follow this exact schema: "
    '{"id":"...","age_group":"...","gender":"...",'
    '"upper_garment":{"type":"...","color":"...","length":"..."},'
    '"lower_garment":{"type":"...","color":"...","length":"..."},'
    '"shoes":{"type":"...","color":"..."},'
    '"head":{"accessories":"...","hairstyle":"...","hair_color":"...","hair_length":"..."},'
    '"carried_items":{"handbag":"...","backpack":"..."},'
    '"handheld_items":{"dangerous_item":"...","mobile_phone":"..."},"extra":[]}. '
    "Use \"unknown\" for every missing scalar field and an empty array for extra. "
    "Do not infer from typical appearance and do not add commentary."
)


def endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def parse_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    return json.loads(cleaned)


def request_batch(
    batch: Sequence[Mapping[str, Any]], *, base_url: str, model: str,
    timeout: float, retries: int, max_tokens: int
) -> List[Dict[str, Any]]:
    items = [{"id": str(row.get("sample_id") or row.get("row_index")), "caption": row.get("prediction", "")} for row in batch]
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
            req = urllib.request.Request(endpoint(base_url), data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}, method="POST")
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


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    if row.get("id") is not None and row.get("error") is None and isinstance(row.get("attributes"), dict):
                        done.add(str(row["id"]))
                except json.JSONDecodeError:
                    continue
    return done


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    pending: List[Dict[str, Any]] = []
    submitted = 0
    completed = 0
    errors = 0
    with args.output.open("a", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures: Dict[concurrent.futures.Future[List[Dict[str, Any]]], Sequence[Mapping[str, Any]]] = {}
            for row in iter_rows(args.input):
                sample_id = str(row.get("sample_id") or row.get("row_index"))
                if sample_id in done:
                    continue
                pending.append(row)
                if len(pending) < args.batch_size:
                    continue
                future = pool.submit(request_batch, tuple(pending), base_url=args.base_url, model=args.model, timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens)
                futures[future] = tuple(pending)
                submitted += len(pending)
                pending = []
                while len(futures) >= args.workers * 2:
                    finished = next(iter(concurrent.futures.as_completed(futures)))
                    source_rows = futures.pop(finished)
                    for result in finished.result():
                        source = next((r for r in source_rows if str(r.get("sample_id") or r.get("row_index")) == result["id"]), None)
                        record = {"id": result["id"], "attributes": result["attributes"], "raw": result["raw"], "error": result["error"]}
                        if source is not None:
                            record.update({"sample_id": source.get("sample_id"), "scene": source.get("scene"), "session": source.get("session"), "prediction": source.get("prediction"), "ground_truth": source.get("attributes")})
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        completed += 1
                        errors += int(result["error"] is not None)
                    out.flush()
            if pending:
                future = pool.submit(request_batch, tuple(pending), base_url=args.base_url, model=args.model, timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens)
                futures[future] = tuple(pending)
                submitted += len(pending)
            for future, source_rows in list(futures.items()):
                for result in future.result():
                    source = next((r for r in source_rows if str(r.get("sample_id") or r.get("row_index")) == result["id"]), None)
                    record = {"id": result["id"], "attributes": result["attributes"], "raw": result["raw"], "error": result["error"]}
                    if source is not None:
                        record.update({"sample_id": source.get("sample_id"), "scene": source.get("scene"), "session": source.get("session"), "prediction": source.get("prediction"), "ground_truth": source.get("attributes")})
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed += 1
                    errors += int(result["error"] is not None)
                out.flush()
    manifest = {"input": str(args.input), "output": str(args.output), "model": args.model, "base_url": args.base_url, "batch_size": args.batch_size, "workers": args.workers, "submitted": submitted, "completed": completed, "errors": errors}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:6097/v1"))
    parser.add_argument("--model", default="Qwen36-35b-caption")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        parser.error("batch-size and workers must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
