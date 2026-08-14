# Three-Checkpoint RL Test Evaluation Design

## Goal

Evaluate the frozen 4,328-sample person test set once for V4B SFT, Qwen RL,
and lexical RL, then publish a compact eight-metric comparison with paired
confidence intervals and a checkpoint recommendation.

## Frozen Inputs

- Test data: `data/prepared/test.jsonl`; record its SHA-256 and require 4,328
  unique `sample_id` values in identical order for all candidates.
- V4B SFT: `artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final`.
- Qwen RL: `artifacts/rl/region_category_person_scst_reviewed5981_qwen4b/final`.
- Lexical RL: `artifacts/rl/region_category_person_scst_reviewed5981_lexical/final`.
- Florence decoding: greedy, `do_sample=false`, `num_beams=1`,
  `max_new_tokens=64`, using each checkpoint's own saved processor.
- Attribute extractor: Qwen3.6-27B-FP8 at `http://127.0.0.1:6097/v1`, thinking
  disabled. It converts every candidate caption to the same canonical schema.
- Semantic judge: Qwen3.5-4B at `http://127.0.0.1:6098/v1`, thinking disabled,
  using the frozen five-level similarity prompt from `scripts/rl_reward.py`.

The hand-reviewed `attributes` object in each test row is the common ground
truth. Historical V4B extraction metrics are not reused because they were
produced by an older service configuration. Existing V4B Florence predictions
may be reused only after checkpoint path, sample count, sample IDs, and decoding
configuration pass validation.

## Eight Metric Groups

1. **Lexical Micro P/R/F1**: aggregate soft TP/FP/FN across all known scalar
   fields and `extra`, using the frozen lexical similarity function.
2. **Qwen Semantic Micro P/R/F1**: use the same extracted attributes and ground
   truth, replacing only lexical value similarity with the frozen Qwen4B
   five-level judge.
3. **Lexical Macro-field F1**: arithmetic mean of lexical soft F1 for the 18
   scalar fields plus `extra`; fields with no GT positives in the complete test
   set are excluded.
4. **Fabrication ratio**: total fabricated assertions divided by total generated
   assertions. Fabrication means a concrete prediction in a GT-unknown scalar
   slot or an unmatched generated `extra` item.
5. **Structure pass rate**: percentage satisfying both single-sentence output
   and the frozen person-subject-start regular expression.
6. **Length**: report mean word count and the percentage within 18-24 words.
7. **Background leakage rate**: percentage containing a frozen background term.
8. **Invalid output rate**: percentage that is empty, JSON-like, pipe-delimited,
   or contains residual special tokens.

Lexical and semantic F1 reuse identical prediction extraction and GT data, so
the comparison isolates matching strictness rather than extraction variance.

## Scoring And Selection

The primary ranking value is `0.5 * lexical_micro_f1 +
0.5 * qwen_semantic_micro_f1`, but it is applied only after gates:

- neither Micro F1 may fall more than 1.0 absolute percentage point below V4B;
- fabrication may not be significantly worse than V4B;
- structure, background, invalid-output, and length metrics may not materially
  regress from V4B or violate the experiment-plan release thresholds.

Ties are resolved by lower fabrication, higher semantic recall, higher lexical
macro-field F1, then better 18-24-word compliance. A difference at or below 0.5
percentage point, or whose paired session-bootstrap 95% confidence interval
includes zero, is described as no clear improvement; if all candidates tie,
retain V4B.

## Statistical And Diagnostic Output

Use deterministic paired bootstrap resampling by `session`, with the same
resampled sessions for all three candidates and 2,000 replicates. Report each
absolute metric, delta from V4B, and the delta's 95% percentile interval.

For the recommendation explanation, include per-sample win/tie/loss counts for
both F1 methods and the five fields with the largest gain and loss in soft
TP/FN/FP relative to V4B. These diagnostics explain the choice but do not add
headline metrics.

## Components And Data Flow

- `scripts/rl_test_metrics.py`: import-safe pure scoring, aggregation,
  bootstrap, ranking, and Markdown/JSON report generation.
- `scripts/evaluate_person_attribute_rl.py`: validates inputs, runs resumable
  Qwen extraction and semantic judging, writes per-sample score details, then
  calls the pure report layer.
- `scripts/run_person_attribute_rl_evaluation.sh`: waits for all final
  checkpoints, performs/reuses validated Florence inference serially on GPU
  3-6, evaluates candidates serially against services on GPU 1-2, and resumes
  from completed artifacts after interruption.
- `tests/test_rl_test_metrics.py` and
  `tests/test_evaluate_person_attribute_rl.py`: cover metric definitions,
  order preservation, resume validation, bootstrap determinism, gates, and
  recommendation behavior.

Artifacts live under `artifacts/rl/evaluation/`: one directory per candidate,
plus `person_caption_metrics.json`, `person_caption_comparison.md`, a run-state
manifest, per-sample details, and field deltas. Writes are atomic; completion
markers are accepted only after hashes, counts, and ordered IDs validate.

## Failure Handling

Extraction and judge calls are bounded-concurrency, retryable, and resumable.
Failed extraction rows are retried individually; unresolved failures stop final
publication instead of being silently counted as model errors. Judge requests
are cached by `(field, gt, prediction)` and stored persistently. The report is
not published unless all three candidates have exactly 4,328 ordered rows and
zero unresolved service failures.

## Self-Review

The design fixes all model paths, test identity, service versions, metric
formulas, ranking gates, confidence method, output locations, and failure
semantics. It intentionally excludes general COCO retention and human review
from this compact RL-selection run because the approved scope is the eight
test-set metrics needed to distinguish V4B, Qwen RL, and lexical RL.
