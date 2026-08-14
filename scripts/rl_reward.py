#!/usr/bin/env python3
"""Pure RL reward logic for Florence2 person-attribute SCST.

Merged F1-core design (EXPERIMENT_PLAN.md §11.2): a soft attribute F1 (matches /
conflicts / misses) is the main term, a fabrication penalty covers F1's blind
spot (GT=unknown but generated, plus unmatched extra), and structure/length/
background are small format terms. The similarity function used inside the soft
F1 is injected, so two backends share the same code:

  - lexical_similarity : value_similarity (token overlap), mirrors the evaluator
  - Qwen 5-level judge  : {1.0, 0.75, 0.5, 0.25, 0.0} (semantic, see rl_clients)

Import-safe: no torch, no network.
"""

import re
from typing import Any, Callable, Dict, Mapping

from evaluate_qwen_attribute_extraction import (
    BACKGROUND,
    SCALAR_FIELDS,
    SUBJECT_START,
    normalize_extra,
    normalize_value,
    normalize_value_set,
    value_similarity,
)


# Reward weights. F1 (attribute correctness) and fabrication (多说, F1's blind
# spot) are the two equal main terms (0.40); structure/length/background are
# small format terms (0.10). Length is signed (ideal 18-24 words rewarded).
W_F1 = 0.40
W_FABRICATION = 0.40
W_STRUCTURE = 0.10
W_LENGTH = 0.10
W_BACKGROUND = 0.10
EXTRA_MATCH_THRESHOLD = 0.5


JUDGE_PROMPT = """You are a strict attribute-value similarity scorer for person appearance descriptions.

You receive ONE attribute field, a ground-truth (GT) value, and a generated value
(extracted from a caption). Score their similarity on a 5-point scale.

Scores:
- 1.00  exact match or synonym / different specificity of the same value
        "white" vs "light-colored" | "dark" vs "black" | "navy" vs "dark blue"
        "bright red" vs "red" | "middle-aged" vs "adult" | "shorts" vs "short pants"
- 0.75  strongly related (same category, very close)
        "jacket" vs "coat" | "pants" vs "jeans"
- 0.50  partial: a generic term weakly covering a specific one (fallback)
        "top" vs "jacket" | "top" vs "t-shirt" | "shoes" vs "sneakers"
- 0.25  weakly related (same broad category, clearly different)
        "shirt" vs "jacket" | "blue" vs "purple"
- 0.00  conflict: genuinely different value, even if same kind of attribute
        "blue" vs "red" | "male" vs "female"
        "short-sleeved" vs "long-sleeved" | "short hair" vs "long hair"

Never score above 0.00 for two values that name different things just because
they share a category. Same category but different value = 0.00.

Field: {field}
GT value: {gt_value}
Generated value: {gen_value}

Respond with ONLY this JSON object, no other text:
{{"score": 1.0 | 0.75 | 0.5 | 0.25 | 0.0, "reason": "<one short clause>"}}
"""


_PIPE_FIELDS = re.compile(r"\b\w+\|\w+")


def _nested(attributes: Mapping[str, Any], path: str) -> Any:
    value: Any = attributes
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def length_score(word_count: int) -> float:
    """Reward 18-24 words, lightly penalize 12-17/25-28, else penalize fully."""
    if 18 <= word_count <= 24:
        return 1.0
    if (12 <= word_count <= 17) or (25 <= word_count <= 28):
        return -0.5
    return -1.0


def _is_single_sentence(caption: str) -> bool:
    return len(re.findall(r"[.!?]+", caption)) <= 1


def sentence_structure_score(caption: str) -> float:
    """1.0 for a single natural sentence with a person subject near the start."""
    if not caption.strip():
        return 0.0
    if _is_single_sentence(caption) and bool(SUBJECT_START.match(caption)):
        return 1.0
    return 0.0


def background_penalty(caption: str) -> float:
    """1.0 if the caption mentions scene/building/location/background, else 0.0."""
    return 1.0 if BACKGROUND.search(caption) else 0.0


def is_invalid_format(caption: str) -> bool:
    """True for empty output, JSON, or pipe-delimited fields (hard reward floor)."""
    text = caption.strip()
    if not text:
        return True
    if text[0] in "{[":
        return True
    if _PIPE_FIELDS.search(text):
        return True
    return False


def lexical_similarity(field: str, gt_value: str, gen_value: str) -> float:
    """Lexical similarity (token overlap), identical to the evaluator's soft F1."""
    return value_similarity(gt_value, gen_value)


def classify_sample(
    gt_attrs: Mapping[str, Any],
    gen_attrs: Mapping[str, Any],
    similarity_fn: Callable[[str, str, str], float] = lexical_similarity,
) -> Dict[str, Any]:
    """Soft-classify each fixed field and extra item against GT.

    Returns counts consumed by compute_reward:
      soft_tp / soft_fp / soft_fn : similarity-weighted soft-F1 counts
      n_gen_assert : total concrete generated assertions (fixed + extra)
      n_fab        : fabrications (GT=unknown but generated, + unmatched extra)
    """
    soft_tp = 0.0
    soft_fp = 0.0
    soft_fn = 0.0
    n_gen_assert = 0
    n_fab = 0
    for field in SCALAR_FIELDS:
        gt_values = normalize_value_set(field, _nested(gt_attrs, field))
        gen_value = normalize_value(field, _nested(gen_attrs, field))
        if not gt_values and gen_value is None:
            continue
        if not gt_values and gen_value is not None:
            n_fab += 1          # 多说 on an unknown GT field (F1 blind spot)
            n_gen_assert += 1
            continue
        if gen_value is None:
            soft_fn += 1.0      # 少说 (miss)
            continue
        n_gen_assert += 1
        s = max(float(similarity_fn(field, g, gen_value)) for g in gt_values)
        soft_tp += s
        soft_fp += 1.0 - s
        soft_fn += 1.0 - s

    gt_extra = normalize_extra(gt_attrs.get("extra"))
    gen_extra = normalize_extra(gen_attrs.get("extra"))
    n_gen_assert += len(gen_extra)
    unmatched = list(range(len(gen_extra)))
    for gt_value in gt_extra:
        best_idx = None
        best_s = 0.0
        for idx in unmatched:
            s = float(similarity_fn("extra", gt_value, gen_extra[idx]))
            if s > best_s:
                best_s, best_idx = s, idx
        if best_idx is not None and best_s >= EXTRA_MATCH_THRESHOLD:
            unmatched.remove(best_idx)
            soft_tp += best_s
            soft_fp += 1.0 - best_s
            soft_fn += 1.0 - best_s
        else:
            soft_fn += 1.0      # GT extra not covered (recall miss)
    n_fab += len(unmatched)     # unmatched generated extra = fabrication

    return {
        "soft_tp": soft_tp,
        "soft_fp": soft_fp,
        "soft_fn": soft_fn,
        "n_gen_assert": n_gen_assert,
        "n_fab": n_fab,
    }


def compute_reward(caption: str, counts: Mapping[str, Any]) -> float:
    """Full SCST reward in [-1, 1]. Empty / invalid format -> -1.0."""
    if is_invalid_format(caption):
        return -1.0
    soft_tp = float(counts["soft_tp"])
    soft_fp = float(counts["soft_fp"])
    soft_fn = float(counts["soft_fn"])
    n_gen_assert = counts["n_gen_assert"]
    n_fab = counts["n_fab"]

    denom = 2.0 * soft_tp + soft_fp + soft_fn
    f1 = (2.0 * soft_tp / denom) if denom > 0 else 0.0
    fabrication_ratio = n_fab / max(n_gen_assert, 1)

    reward = (
        W_F1 * f1
        - W_FABRICATION * fabrication_ratio
        + W_STRUCTURE * sentence_structure_score(caption)
        + W_LENGTH * length_score(len(caption.split()))
        - W_BACKGROUND * background_penalty(caption)
    )
    return max(-1.0, min(1.0, reward))


def reward_for_sample(caption: str, gt_attrs: Mapping[str, Any], gen_attrs: Mapping[str, Any],
                      similarity_fn: Callable[[str, str, str], float] = lexical_similarity) -> float:
    """Convenience: classify then score, given already-extracted gen_attrs."""
    return compute_reward(caption, classify_sample(gt_attrs, gen_attrs, similarity_fn))
