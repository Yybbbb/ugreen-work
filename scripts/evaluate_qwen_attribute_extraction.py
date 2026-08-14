#!/usr/bin/env python3
"""Deduplicate Qwen attribute extraction and compute reproducible test metrics."""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCALAR_FIELDS = (
    "age_group", "gender",
    "upper_garment.type", "upper_garment.color", "upper_garment.length",
    "lower_garment.type", "lower_garment.color", "lower_garment.length",
    "shoes.type", "shoes.color",
    "head.accessories", "head.hairstyle", "head.hair_color", "head.hair_length",
    "carried_items.handbag", "carried_items.backpack",
    "handheld_items.dangerous_item", "handheld_items.mobile_phone",
)
ALL_FIELDS = SCALAR_FIELDS + ("extra",)
UNKNOWN = {"", "unknown", "none", "no", "null", "n/a", "na", "not applicable", "unspecified"}
SUBJECT_START = re.compile(
    r"^(?:a|an|the)\s+(?:adult|young|elderly|older|middle-aged|child|teenage|male|female|man|woman|person|boy|girl)\b",
    re.I,
)
BACKGROUND = re.compile(
    r"\b(?:background|scene|street|road|building|room|store|shop|office|park|indoor|outdoor|wall|vehicle|camera)\b",
    re.I,
)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
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


def deduplicate_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        row_id = str(row.get("id") or row.get("sample_id") or "")
        if not row_id:
            continue
        success = row.get("error") is None and isinstance(row.get("attributes"), dict)
        current = selected.get(row_id)
        current_success = current is not None and current.get("error") is None and isinstance(current.get("attributes"), dict)
        if success or not current_success:
            selected[row_id] = row
    return selected


def _nested(attributes: Mapping[str, Any], path: str) -> Any:
    value: Any = attributes
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def normalize_value(field: str, value: Any) -> Optional[str]:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip().lower().replace("gray", "grey")
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[-]+", "-", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    if text in UNKNOWN:
        return None
    if field == "gender":
        text = {"man": "male", "boy": "male", "woman": "female", "girl": "female"}.get(text, text)
    if field.endswith(".length"):
        compact = text.replace("-", " ")
        text = {
            "long sleeved": "long-sleeve", "long sleeve": "long-sleeve",
            "short sleeved": "short-sleeve", "short sleeve": "short-sleeve",
            "sleeveless": "sleeveless",
        }.get(compact, text)
    if field.endswith(".type"):
        text = text.replace("-", " ")
        text = re.sub(r"^(?:long|short) sleeve(?:d)?\s+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = {
            "t shirt": "t-shirt", "tee": "t-shirt", "tee shirt": "t-shirt",
            "trousers": "pants", "trainer": "sneakers", "trainers": "sneakers",
            "sport shoes": "sneakers", "athletic shoes": "sneakers",
        }.get(text, text)
    if field == "head.accessories":
        text = text.replace("eyeglasses", "glasses")
    return text


def normalize_value_set(field: str, value: Any) -> List[str]:
    """All acceptable normalized values for a field, list-aware.

    Mirrors ``normalize_value`` but accepts list-valued GT (e.g. ``['black',
    'white']`` for a two-tone garment) by normalizing every element, dropping
    unknowns, and deduplicating while preserving order. A scalar value yields a
    single-element list (or empty when unknown), so callers can uniformly take
    ``max(similarity(g, pred) for g in gt_values)``. Predictions are assumed
    scalar (as produced by the extractor) and stay on ``normalize_value``.
    """
    if isinstance(value, list):
        kept: List[str] = []
        for item in value:
            normalized = normalize_value(field, item)
            if normalized is not None and normalized not in kept:
                kept.append(normalized)
        return kept
    normalized = normalize_value(field, value)
    return [] if normalized is None else [normalized]


def normalize_extra(value: Any) -> List[str]:
    values = value if isinstance(value, list) else ([] if value is None else [value])
    normalized = []
    for item in values:
        kept = normalize_value("extra", item)
        if kept and kept not in normalized:
            normalized.append(kept)
    return normalized


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value) if token not in {"a", "an", "the", "with", "visible", "wearing", "holding"}}


def value_similarity(left: Optional[str], right: Optional[str]) -> float:
    if left is None or right is None:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.9
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def _empty_counts() -> Dict[str, float]:
    return {"tp": 0, "fp": 0, "fn": 0, "soft_tp": 0.0, "soft_fp": 0.0, "soft_fn": 0.0, "known": 0, "exact": 0, "unknown_slots": 0, "unknown_extra": 0}


def score_attribute_pair(gt: Mapping[str, Any], pred: Mapping[str, Any]) -> Dict[str, float]:
    counts = _empty_counts()
    for field in SCALAR_FIELDS:
        gt_values = normalize_value_set(field, _nested(gt, field))
        pred_value = normalize_value(field, _nested(pred, field))
        if not gt_values:
            counts["unknown_slots"] += 1
            counts["unknown_extra"] += int(pred_value is not None)
            continue
        counts["known"] += 1
        if pred_value is None:
            counts["fn"] += 1
            counts["soft_fn"] += 1.0
            continue
        similarity = max(value_similarity(g, pred_value) for g in gt_values)
        counts["soft_tp"] += similarity
        counts["soft_fp"] += 1.0 - similarity
        counts["soft_fn"] += 1.0 - similarity
        if pred_value in gt_values:
            counts["tp"] += 1
            counts["exact"] += 1
        else:
            counts["fp"] += 1
            counts["fn"] += 1

    gt_extra = normalize_extra(gt.get("extra"))
    pred_extra = normalize_extra(pred.get("extra"))
    unmatched = set(range(len(pred_extra)))
    for gt_value in gt_extra:
        best_index = None
        best_similarity = 0.0
        for index in unmatched:
            similarity = value_similarity(gt_value, pred_extra[index])
            if similarity > best_similarity:
                best_similarity, best_index = similarity, index
        counts["known"] += 1
        if best_index is not None and best_similarity >= 0.65:
            unmatched.remove(best_index)
            counts["soft_tp"] += best_similarity
            counts["soft_fp"] += 1.0 - best_similarity
            counts["soft_fn"] += 1.0 - best_similarity
            if gt_value == pred_extra[best_index]:
                counts["tp"] += 1
                counts["exact"] += 1
            else:
                counts["fp"] += 1
                counts["fn"] += 1
        else:
            counts["fn"] += 1
            counts["soft_fn"] += 1.0
    counts["fp"] += len(unmatched)
    counts["soft_fp"] += len(unmatched)
    if not gt_extra:
        counts["unknown_slots"] += 1
        counts["unknown_extra"] += len(pred_extra)
    return counts


def _prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def evaluate(rows: Sequence[Mapping[str, Any]], input_lines: int) -> Dict[str, Any]:
    total = _empty_counts()
    per_field = {field: _empty_counts() for field in ALL_FIELDS}
    words: List[int] = []
    empty = single = subject = background = special = failures = 0
    for row in rows:
        if row.get("error") is not None or not isinstance(row.get("attributes"), Mapping):
            failures += 1
            continue
        gt = row.get("ground_truth") if isinstance(row.get("ground_truth"), Mapping) else {}
        pred = row["attributes"]
        pair = score_attribute_pair(gt, pred)
        for key in total:
            total[key] += pair[key]
        for field in SCALAR_FIELDS:
            field_gt = {}
            field_pred = {}
            gt_value = _nested(gt, field)
            pred_value = _nested(pred, field)
            target_gt, target_pred = field_gt, field_pred
            parts = field.split(".")
            for part in parts[:-1]:
                target_gt[part] = {}; target_gt = target_gt[part]
                target_pred[part] = {}; target_pred = target_pred[part]
            target_gt[parts[-1]] = gt_value; target_pred[parts[-1]] = pred_value
            scored = _empty_counts()
            g = normalize_value(field, gt_value); p = normalize_value(field, pred_value)
            if g is None:
                scored["unknown_slots"] = 1; scored["unknown_extra"] = int(p is not None)
            else:
                scored["known"] = 1
                if p is None:
                    scored["fn"] = 1; scored["soft_fn"] = 1.0
                else:
                    sim = value_similarity(g, p); scored["soft_tp"] = sim; scored["soft_fp"] = 1-sim; scored["soft_fn"] = 1-sim
                    if g == p: scored["tp"] = 1; scored["exact"] = 1
                    else: scored["fp"] = 1; scored["fn"] = 1
            for key in scored: per_field[field][key] += scored[key]
        caption = str(row.get("prediction") or "").strip()
        count = len(caption.split()); words.append(count)
        empty += int(not caption); single += int(bool(caption) and len(re.findall(r"[.!?]+", caption)) <= 1)
        subject += int(bool(SUBJECT_START.match(caption))); background += int(bool(BACKGROUND.search(caption)))
        special += int(any(token in caption for token in ("<pad>", "</s>", "<s>")))
    micro = _prf(total["tp"], total["fp"], total["fn"])
    soft = _prf(total["soft_tp"], total["soft_fp"], total["soft_fn"])
    field_metrics = {}
    f1_values = []
    for field in SCALAR_FIELDS:
        metric = _prf(per_field[field]["tp"], per_field[field]["fp"], per_field[field]["fn"])
        metric.update({"known": per_field[field]["known"], "exact": per_field[field]["exact"] / per_field[field]["known"] if per_field[field]["known"] else 0.0})
        field_metrics[field] = metric
        if per_field[field]["known"]:
            f1_values.append(metric["f1"])
    valid = len(rows) - failures
    sorted_words = sorted(words)
    p95 = sorted_words[min(len(sorted_words)-1, math.ceil(len(sorted_words)*0.95)-1)] if sorted_words else 0
    return {
        "samples": len(rows), "input_lines": input_lines, "duplicate_lines": input_lines-len(rows), "extractor_failures": failures,
        "micro": micro, "soft_micro": soft,
        "macro_field_f1": sum(f1_values)/len(f1_values) if f1_values else 0.0,
        "mean_field_exact": total["exact"]/total["known"] if total["known"] else 0.0,
        "known_attributes": int(total["known"]),
        "unknown_slots": int(total["unknown_slots"]),
        "unknown_extra_count": int(total["unknown_extra"]),
        "unknown_extra_rate": total["unknown_extra"]/total["unknown_slots"] if total["unknown_slots"] else 0.0,
        "caption": {
            "average_words": sum(words)/len(words) if words else 0.0, "p95_words": p95,
            "length_18_24_ratio": sum(18 <= value <= 24 for value in words)/len(words) if words else 0.0,
            "empty_ratio": empty/valid if valid else 0.0, "single_sentence_ratio": single/valid if valid else 0.0,
            "subject_start_ratio": subject/valid if valid else 0.0, "background_keyword_ratio": background/valid if valid else 0.0,
            "special_token_ratio": special/valid if valid else 0.0,
        },
        "per_field": field_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_rows = list(iter_jsonl(args.input))
    dedup = deduplicate_rows(source_rows)
    rows = [dedup[key] for key in sorted(dedup)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(args.output_dir / "qwen_attributes.dedup.jsonl", rows)
    metrics = evaluate(rows, len(source_rows))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
