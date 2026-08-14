#!/usr/bin/env python3
"""Pure metrics and comparison logic for the three-checkpoint RL test."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from evaluate_qwen_attribute_extraction import (
    ALL_FIELDS,
    BACKGROUND,
    SCALAR_FIELDS,
    SUBJECT_START,
    normalize_extra,
    normalize_value,
    normalize_value_set,
    value_similarity,
)
from rl_reward import EXTRA_MATCH_THRESHOLD, is_invalid_format


SimilarityFn = Callable[[str, str, str], float]
SPECIAL_TOKENS = ("<pad>", "</s>", "<s>")
HEADLINE_METRIC_LABELS = (
    "Lexical Micro P/R/F1",
    "Qwen Semantic Micro P/R/F1",
    "Lexical Macro-field F1",
    "Fabrication ratio",
    "Structure pass rate",
    "Length mean / 18-24 ratio",
    "Background leakage rate",
    "Invalid output rate",
)


def lexical_similarity(field: str, gt_value: str, pred_value: str) -> float:
    return value_similarity(gt_value, pred_value)


def _nested(attributes: Mapping[str, Any], path: str) -> Any:
    value: Any = attributes
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _empty_counts() -> Dict[str, float]:
    return {
        "soft_tp": 0.0,
        "soft_fp": 0.0,
        "soft_fn": 0.0,
        "n_gen_assert": 0.0,
        "n_fab": 0.0,
        "gt_positive": 0.0,
    }


def _add_counts(target: Dict[str, float], source: Mapping[str, float]) -> None:
    for key in target:
        target[key] += float(source.get(key, 0.0))


def _prf(counts: Mapping[str, float]) -> Dict[str, float]:
    tp = float(counts["soft_tp"])
    fp = float(counts["soft_fp"])
    fn = float(counts["soft_fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_attribute_pair(
    ground_truth: Mapping[str, Any],
    prediction: Mapping[str, Any],
    similarity_fn: SimilarityFn,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    total = _empty_counts()
    fields = {field: _empty_counts() for field in ALL_FIELDS}
    for field in SCALAR_FIELDS:
        counts = fields[field]
        gt_values = normalize_value_set(field, _nested(ground_truth, field))
        pred_value = normalize_value(field, _nested(prediction, field))
        if pred_value is not None:
            counts["n_gen_assert"] += 1.0
        if not gt_values:
            if pred_value is not None:
                counts["n_fab"] += 1.0
            _add_counts(total, counts)
            continue
        counts["gt_positive"] += 1.0
        if pred_value is None:
            counts["soft_fn"] += 1.0
        else:
            similarity = min(
                1.0,
                max(0.0, max(float(similarity_fn(field, g, pred_value)) for g in gt_values)),
            )
            counts["soft_tp"] += similarity
            counts["soft_fp"] += 1.0 - similarity
            counts["soft_fn"] += 1.0 - similarity
        _add_counts(total, counts)

    extra_counts = fields["extra"]
    gt_extra = normalize_extra(ground_truth.get("extra"))
    pred_extra = normalize_extra(prediction.get("extra"))
    extra_counts["gt_positive"] = float(len(gt_extra))
    extra_counts["n_gen_assert"] = float(len(pred_extra))
    unmatched = list(range(len(pred_extra)))
    for gt_value in gt_extra:
        best_index: Optional[int] = None
        best_similarity = 0.0
        for index in unmatched:
            similarity = min(
                1.0,
                max(0.0, float(similarity_fn("extra", gt_value, pred_extra[index]))),
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        if best_index is not None and best_similarity >= EXTRA_MATCH_THRESHOLD:
            unmatched.remove(best_index)
            extra_counts["soft_tp"] += best_similarity
            extra_counts["soft_fp"] += 1.0 - best_similarity
            extra_counts["soft_fn"] += 1.0 - best_similarity
        else:
            extra_counts["soft_fn"] += 1.0
    extra_counts["n_fab"] = float(len(unmatched))
    _add_counts(total, extra_counts)
    return total, fields


def _caption_stats(caption: str) -> Dict[str, float]:
    text = caption.strip()
    word_count = len(text.split())
    special = any(token in text for token in SPECIAL_TOKENS)
    single_sentence = bool(text) and len(re.findall(r"[.!?]+", text)) <= 1
    subject_start = bool(SUBJECT_START.match(text))
    return {
        "word_count": float(word_count),
        "length_18_24": float(18 <= word_count <= 24),
        "structure_pass": float(single_sentence and subject_start),
        "background_leakage": float(bool(BACKGROUND.search(text))),
        "invalid_output": float(is_invalid_format(text) or special),
    }


def score_rows(
    rows: Sequence[Mapping[str, Any]], similarity_fn: SimilarityFn
) -> Dict[str, Any]:
    total = _empty_counts()
    per_field = {field: _empty_counts() for field in ALL_FIELDS}
    samples: List[Dict[str, Any]] = []
    caption_totals = {
        "word_count": 0.0,
        "length_18_24": 0.0,
        "structure_pass": 0.0,
        "background_leakage": 0.0,
        "invalid_output": 0.0,
    }
    for row in rows:
        gt = row.get("ground_truth") if isinstance(row.get("ground_truth"), Mapping) else {}
        pred = row.get("attributes") if isinstance(row.get("attributes"), Mapping) else {}
        counts, field_counts = score_attribute_pair(gt, pred, similarity_fn)
        caption = _caption_stats(str(row.get("prediction") or ""))
        _add_counts(total, counts)
        for field in ALL_FIELDS:
            _add_counts(per_field[field], field_counts[field])
        for key in caption_totals:
            caption_totals[key] += caption[key]
        samples.append({
            "sample_id": str(row.get("sample_id") or ""),
            "session": str(row.get("session") or ""),
            "counts": counts,
            "per_field": field_counts,
            "caption": caption,
            "f1": _prf(counts)["f1"],
        })

    count = len(rows)
    field_metrics: Dict[str, Dict[str, float]] = {}
    macro_values: List[float] = []
    for field in ALL_FIELDS:
        metric = _prf(per_field[field])
        metric["gt_positive"] = per_field[field]["gt_positive"]
        field_metrics[field] = metric
        if per_field[field]["gt_positive"] > 0:
            macro_values.append(metric["f1"])
    metrics = {
        "samples": count,
        "micro": _prf(total),
        "macro_field_f1": sum(macro_values) / len(macro_values) if macro_values else 0.0,
        "fabrication_ratio": total["n_fab"] / max(total["n_gen_assert"], 1.0),
        "structure_pass_rate": caption_totals["structure_pass"] / count if count else 0.0,
        "length": {
            "mean_words": caption_totals["word_count"] / count if count else 0.0,
            "length_18_24_ratio": caption_totals["length_18_24"] / count if count else 0.0,
        },
        "background_leakage_rate": caption_totals["background_leakage"] / count if count else 0.0,
        "invalid_output_rate": caption_totals["invalid_output"] / count if count else 0.0,
        "per_field": field_metrics,
        "counts": total,
    }
    return {"metrics": metrics, "samples": samples}


def build_candidate(
    name: str,
    checkpoint: str,
    rows: Sequence[Mapping[str, Any]],
    semantic_similarity_fn: SimilarityFn,
) -> Dict[str, Any]:
    """Score one candidate twice while sharing captions, GT and extraction."""
    lexical = score_rows(rows, lexical_similarity)
    semantic = score_rows(rows, semantic_similarity_fn)
    return {
        "name": name,
        "checkpoint": checkpoint,
        "lexical": lexical,
        "semantic": semantic,
    }


def _aggregate_scored_samples(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = _empty_counts()
    per_field = {field: _empty_counts() for field in ALL_FIELDS}
    captions = {
        "word_count": 0.0,
        "length_18_24": 0.0,
        "structure_pass": 0.0,
        "background_leakage": 0.0,
        "invalid_output": 0.0,
    }
    for sample in samples:
        _add_counts(total, sample["counts"])
        for field in ALL_FIELDS:
            _add_counts(per_field[field], sample["per_field"][field])
        for key in captions:
            captions[key] += float(sample["caption"][key])
    macro = [
        _prf(per_field[field])["f1"]
        for field in ALL_FIELDS
        if per_field[field]["gt_positive"] > 0
    ]
    count = len(samples)
    return {
        "micro": _prf(total),
        "macro_field_f1": sum(macro) / len(macro) if macro else 0.0,
        "fabrication_ratio": total["n_fab"] / max(total["n_gen_assert"], 1.0),
        "structure_pass_rate": captions["structure_pass"] / count if count else 0.0,
        "length": {
            "mean_words": captions["word_count"] / count if count else 0.0,
            "length_18_24_ratio": captions["length_18_24"] / count if count else 0.0,
        },
        "background_leakage_rate": captions["background_leakage"] / count if count else 0.0,
        "invalid_output_rate": captions["invalid_output"] / count if count else 0.0,
        "counts": total,
        "per_field": {field: _prf(per_field[field]) for field in ALL_FIELDS},
    }


def _headline(candidate: Mapping[str, Any], sample_indices: Optional[Sequence[int]] = None) -> Dict[str, float]:
    if sample_indices is None:
        lexical = candidate["lexical"]["metrics"]
        semantic = candidate["semantic"]["metrics"]
    else:
        lexical_samples = [candidate["lexical"]["samples"][index] for index in sample_indices]
        semantic_samples = [candidate["semantic"]["samples"][index] for index in sample_indices]
        lexical = _aggregate_scored_samples(lexical_samples)
        semantic = _aggregate_scored_samples(semantic_samples)
    return {
        "lexical_precision": lexical["micro"]["precision"],
        "lexical_recall": lexical["micro"]["recall"],
        "lexical_micro_f1": lexical["micro"]["f1"],
        "qwen_precision": semantic["micro"]["precision"],
        "qwen_recall": semantic["micro"]["recall"],
        "qwen_micro_f1": semantic["micro"]["f1"],
        "lexical_macro_field_f1": lexical["macro_field_f1"],
        "fabrication_ratio": lexical["fabrication_ratio"],
        "structure_pass_rate": lexical["structure_pass_rate"],
        "mean_words": lexical["length"]["mean_words"],
        "length_18_24_ratio": lexical["length"]["length_18_24_ratio"],
        "background_leakage_rate": lexical["background_leakage_rate"],
        "invalid_output_rate": lexical["invalid_output_rate"],
        "joint_attribute_f1": 0.5 * (
            lexical["micro"]["f1"] + semantic["micro"]["f1"]
        ),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_route_metrics(
    samples: Sequence[Mapping[str, Any]],
    session_positions: Mapping[str, int],
    weights: Any,
) -> Dict[str, Any]:
    """Vectorize bootstrap aggregation after one sufficient-stat pass per session."""
    import numpy as np

    count_keys = tuple(_empty_counts())
    caption_keys = (
        "word_count", "length_18_24", "structure_pass",
        "background_leakage", "invalid_output",
    )
    session_count = len(session_positions)
    totals = np.zeros((session_count, len(count_keys)), dtype=np.float64)
    fields = np.zeros((session_count, len(ALL_FIELDS), len(count_keys)), dtype=np.float64)
    captions = np.zeros((session_count, len(caption_keys)), dtype=np.float64)
    sample_counts = np.zeros(session_count, dtype=np.float64)
    for sample in samples:
        position = session_positions[str(sample["session"])]
        totals[position] += [float(sample["counts"][key]) for key in count_keys]
        for field_index, field in enumerate(ALL_FIELDS):
            fields[position, field_index] += [
                float(sample["per_field"][field][key]) for key in count_keys
            ]
        captions[position] += [float(sample["caption"][key]) for key in caption_keys]
        sample_counts[position] += 1.0

    boot_total = weights @ totals
    boot_fields = np.tensordot(weights, fields, axes=(1, 0))
    boot_captions = weights @ captions
    boot_count = weights @ sample_counts
    key_index = {key: index for index, key in enumerate(count_keys)}

    def prf(array):
        tp = array[..., key_index["soft_tp"]]
        fp = array[..., key_index["soft_fp"]]
        fn = array[..., key_index["soft_fn"]]
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )
        return precision, recall, f1

    precision, recall, f1 = prf(boot_total)
    unused_field_precision, unused_field_recall, field_f1 = prf(boot_fields)
    included = boot_fields[..., key_index["gt_positive"]] > 0
    macro = np.divide(
        (field_f1 * included).sum(axis=1),
        included.sum(axis=1),
        out=np.zeros(boot_total.shape[0], dtype=np.float64),
        where=included.sum(axis=1) != 0,
    )
    generated = boot_total[:, key_index["n_gen_assert"]]
    fabrication = boot_total[:, key_index["n_fab"]] / np.maximum(generated, 1.0)
    caption_index = {key: index for index, key in enumerate(caption_keys)}
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_field_f1": macro,
        "fabrication_ratio": fabrication,
        "structure_pass_rate": boot_captions[:, caption_index["structure_pass"]] / boot_count,
        "mean_words": boot_captions[:, caption_index["word_count"]] / boot_count,
        "length_18_24_ratio": boot_captions[:, caption_index["length_18_24"]] / boot_count,
        "background_leakage_rate": boot_captions[:, caption_index["background_leakage"]] / boot_count,
        "invalid_output_rate": boot_captions[:, caption_index["invalid_output"]] / boot_count,
    }


def _bootstrap_headlines(
    candidate: Mapping[str, Any],
    session_positions: Mapping[str, int],
    weights: Any,
) -> Dict[str, Any]:
    lexical = _bootstrap_route_metrics(candidate["lexical"]["samples"], session_positions, weights)
    semantic = _bootstrap_route_metrics(candidate["semantic"]["samples"], session_positions, weights)
    return {
        "lexical_precision": lexical["precision"],
        "lexical_recall": lexical["recall"],
        "lexical_micro_f1": lexical["f1"],
        "qwen_precision": semantic["precision"],
        "qwen_recall": semantic["recall"],
        "qwen_micro_f1": semantic["f1"],
        "lexical_macro_field_f1": lexical["macro_field_f1"],
        "fabrication_ratio": lexical["fabrication_ratio"],
        "structure_pass_rate": lexical["structure_pass_rate"],
        "mean_words": lexical["mean_words"],
        "length_18_24_ratio": lexical["length_18_24_ratio"],
        "background_leakage_rate": lexical["background_leakage_rate"],
        "invalid_output_rate": lexical["invalid_output_rate"],
        "joint_attribute_f1": 0.5 * (lexical["f1"] + semantic["f1"]),
    }


def _win_tie_loss(candidate: Mapping[str, Any], baseline: Mapping[str, Any], route: str) -> Dict[str, int]:
    counts = {"win": 0, "tie": 0, "loss": 0}
    for current, base in zip(candidate[route]["samples"], baseline[route]["samples"]):
        delta = float(current["f1"]) - float(base["f1"])
        key = "win" if delta > 1e-12 else ("loss" if delta < -1e-12 else "tie")
        counts[key] += 1
    return counts


def _field_deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> List[Dict[str, float]]:
    result = []
    current = candidate["lexical"]["metrics"]["per_field"]
    base = baseline["lexical"]["metrics"]["per_field"]
    for field in ALL_FIELDS:
        current_counts = _empty_counts()
        base_counts = _empty_counts()
        for sample in candidate["lexical"]["samples"]:
            _add_counts(current_counts, sample["per_field"][field])
        for sample in baseline["lexical"]["samples"]:
            _add_counts(base_counts, sample["per_field"][field])
        result.append({
            "field": field,
            "f1_delta": float(current[field]["f1"]) - float(base[field]["f1"]),
            "soft_tp_delta": current_counts["soft_tp"] - base_counts["soft_tp"],
            "soft_fp_delta": current_counts["soft_fp"] - base_counts["soft_fp"],
            "soft_fn_delta": current_counts["soft_fn"] - base_counts["soft_fn"],
        })
    return sorted(result, key=lambda item: (-abs(item["f1_delta"]), item["field"]))


def compare_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_name: str = "v4b_sft",
    bootstrap_replicates: int = 2000,
    seed: int = 20260720,
) -> Dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    by_name = {str(candidate["name"]): candidate for candidate in candidates}
    if len(by_name) != len(candidates) or baseline_name not in by_name:
        raise ValueError("candidate names must be unique and include the baseline")
    baseline = by_name[baseline_name]
    baseline_samples = baseline["lexical"]["samples"]
    ids = [sample["sample_id"] for sample in baseline_samples]
    sessions: Dict[str, List[int]] = {}
    for index, sample in enumerate(baseline_samples):
        sessions.setdefault(str(sample["session"]), []).append(index)
    session_names = sorted(sessions)
    if not session_names:
        raise ValueError("cannot compare empty candidates")
    for candidate in candidates:
        candidate_ids = [sample["sample_id"] for sample in candidate["lexical"]["samples"]]
        if candidate_ids != ids:
            raise ValueError(f"sample order mismatch for {candidate['name']}")
        semantic_ids = [sample["sample_id"] for sample in candidate["semantic"]["samples"]]
        if semantic_ids != ids:
            raise ValueError(f"semantic sample order mismatch for {candidate['name']}")
        candidate_sessions = [sample["session"] for sample in candidate["lexical"]["samples"]]
        semantic_sessions = [sample["session"] for sample in candidate["semantic"]["samples"]]
        baseline_sessions = [sample["session"] for sample in baseline_samples]
        if candidate_sessions != baseline_sessions or semantic_sessions != baseline_sessions:
            raise ValueError(f"session mismatch for {candidate['name']}")

    import numpy as np

    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(session_names), size=(bootstrap_replicates, len(session_names))
    )
    weights = np.zeros((bootstrap_replicates, len(session_names)), dtype=np.float64)
    np.add.at(
        weights,
        (np.repeat(np.arange(bootstrap_replicates), len(session_names)), draws.ravel()),
        1.0,
    )
    session_positions = {name: index for index, name in enumerate(session_names)}

    absolute = {name: _headline(candidate) for name, candidate in by_name.items()}
    baseline_headline = absolute[baseline_name]
    bootstrap_values = {
        name: _bootstrap_headlines(candidate, session_positions, weights)
        for name, candidate in by_name.items()
    }
    comparisons: Dict[str, Dict[str, Any]] = {}
    for name, candidate in by_name.items():
        if name == baseline_name:
            continue
        delta = {
            key: absolute[name][key] - baseline_headline[key]
            for key in baseline_headline
        }
        distributions = {
            key: (
                bootstrap_values[name][key] - bootstrap_values[baseline_name][key]
            ).tolist()
            for key in baseline_headline
        }
        intervals = {
            key: [_percentile(values, 0.025), _percentile(values, 0.975)]
            for key, values in distributions.items()
        }
        failed_gates = []
        if delta["lexical_micro_f1"] < -0.01:
            failed_gates.append("lexical_micro_f1")
        if delta["qwen_micro_f1"] < -0.01:
            failed_gates.append("qwen_micro_f1")
        if delta["fabrication_ratio"] > 0 and intervals["fabrication_ratio"][0] > 0:
            failed_gates.append("fabrication_ratio")
        if absolute[name]["structure_pass_rate"] < 0.95 or delta["structure_pass_rate"] < -0.01:
            failed_gates.append("structure_pass_rate")
        if absolute[name]["background_leakage_rate"] > 0.01:
            failed_gates.append("background_leakage_rate")
        if delta["invalid_output_rate"] > 0.005:
            failed_gates.append("invalid_output_rate")
        if delta["length_18_24_ratio"] < -0.01:
            failed_gates.append("length_18_24_ratio")
        comparisons[name] = {
            "delta": delta,
            "delta_ci95": intervals,
            "eligible": not failed_gates,
            "failed_gates": failed_gates,
            "win_tie_loss": {
                "lexical": _win_tie_loss(candidate, baseline, "lexical"),
                "qwen_semantic": _win_tie_loss(candidate, baseline, "semantic"),
            },
            "field_deltas": _field_deltas(candidate, baseline),
        }

    eligible = [
        name for name, comparison in comparisons.items() if comparison["eligible"]
    ]
    eligible.sort(key=lambda name: (
        absolute[name]["joint_attribute_f1"],
        -absolute[name]["fabrication_ratio"],
        absolute[name]["qwen_recall"],
        absolute[name]["lexical_macro_field_f1"],
        absolute[name]["length_18_24_ratio"],
    ), reverse=True)
    recommended = baseline_name
    reason = "No eligible RL checkpoint has a clear paired improvement over V4B."
    if eligible:
        best = eligible[0]
        improvement = comparisons[best]["delta"]["joint_attribute_f1"]
        interval = comparisons[best]["delta_ci95"]["joint_attribute_f1"]
        if improvement > 0.005 and interval[0] > 0.0:
            recommended = best
            reason = "Highest eligible joint attribute F1 with a clear paired improvement over V4B."
    return {
        "baseline": baseline_name,
        "bootstrap": {"unit": "session", "replicates": bootstrap_replicates, "seed": seed},
        "candidates": {
            name: {"checkpoint": by_name[name]["checkpoint"], "metrics": absolute[name]}
            for name in by_name
        },
        "comparisons": comparisons,
        "recommendation": {"candidate": recommended, "reason": reason},
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(comparison: Mapping[str, Any]) -> str:
    names = list(comparison["candidates"])
    header = "| Metric | " + " | ".join(names) + " |"
    divider = "|---|" + "---:|" * len(names)
    metrics = {name: comparison["candidates"][name]["metrics"] for name in names}
    rows = [
        (HEADLINE_METRIC_LABELS[0], lambda m: "/".join(_percent(m[key]) for key in ("lexical_precision", "lexical_recall", "lexical_micro_f1"))),
        (HEADLINE_METRIC_LABELS[1], lambda m: "/".join(_percent(m[key]) for key in ("qwen_precision", "qwen_recall", "qwen_micro_f1"))),
        (HEADLINE_METRIC_LABELS[2], lambda m: _percent(m["lexical_macro_field_f1"])),
        (HEADLINE_METRIC_LABELS[3], lambda m: _percent(m["fabrication_ratio"])),
        (HEADLINE_METRIC_LABELS[4], lambda m: _percent(m["structure_pass_rate"])),
        (HEADLINE_METRIC_LABELS[5], lambda m: f"{m['mean_words']:.2f} / {_percent(m['length_18_24_ratio'])}"),
        (HEADLINE_METRIC_LABELS[6], lambda m: _percent(m["background_leakage_rate"])),
        (HEADLINE_METRIC_LABELS[7], lambda m: _percent(m["invalid_output_rate"])),
    ]
    lines = ["# Person Attribute RL Test Comparison", "", header, divider]
    for label, formatter in rows:
        lines.append("| " + label + " | " + " | ".join(formatter(metrics[name]) for name in names) + " |")
    recommendation = comparison["recommendation"]
    lines.extend([
        "",
        "## Delta vs V4B",
        "",
        "| Candidate | Joint F1 delta (95% CI) | Lexical F1 delta | Qwen F1 delta | Fabrication delta | Gates |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for name, item in comparison["comparisons"].items():
        joint_ci = item["delta_ci95"]["joint_attribute_f1"]
        gates = "pass" if item["eligible"] else ", ".join(item["failed_gates"])
        lines.append(
            f"| {name} | {_percent(item['delta']['joint_attribute_f1'])} "
            f"[{_percent(joint_ci[0])}, {_percent(joint_ci[1])}] | "
            f"{_percent(item['delta']['lexical_micro_f1'])} | "
            f"{_percent(item['delta']['qwen_micro_f1'])} | "
            f"{_percent(item['delta']['fabrication_ratio'])} | {gates} |"
        )
    lines.extend([
        "",
        "## Paired Analysis",
        "",
        "Win / tie / loss is counted per test sample against V4B.",
        "",
    ])
    for name, item in comparison["comparisons"].items():
        lexical = item["win_tie_loss"]["lexical"]
        semantic = item["win_tie_loss"]["qwen_semantic"]
        lines.append(
            f"- **{name}**: lexical Win / tie / loss "
            f"{lexical['win']} / {lexical['tie']} / {lexical['loss']}; "
            f"Qwen {semantic['win']} / {semantic['tie']} / {semantic['loss']}."
        )
        changed = [field for field in item["field_deltas"] if abs(field["f1_delta"]) > 1e-12][:5]
        summary = ", ".join(
            f"{field['field']} {field['f1_delta']:+.4f} "
            f"(TP {field['soft_tp_delta']:+.1f}, FP {field['soft_fp_delta']:+.1f}, FN {field['soft_fn_delta']:+.1f})"
            for field in changed
        ) or "no field-level change"
        lines.append(f"- **{name} Largest field changes**: {summary}.")
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"Select **{recommendation['candidate']}**. {recommendation['reason']}",
        "",
    ])
    return "\n".join(lines)
