# Reviewed Person Dataset Merge Design

## Objective

Merge reviewed person annotations into the current dataset under
`/data1/work/MichaelYu/florence-attibute/data` while preserving reviewed
samples, preventing high-caption-similarity leakage between train and test,
and keeping the main split sizes close to their current crop counts.

## Inputs

- Authoritative review records:
  `/data0/work/WangHaoxiang/a35_vehicle_label/outputs/person_reviewed/frames`
- Rewritten reviewed captions:
  `/data1/work/MichaelYu/florence-attibute/data/review`
- Existing dataset:
  `/data1/work/MichaelYu/florence-attibute/data`

Review status, attributes, and dimensions come from the authoritative review
records. A matching caption from `data/review` takes precedence because it has
already been rewritten against the reviewed attributes. If no rewritten crop
matches, use the authoritative reviewed caption.

## Baseline Crop Counts

The per-split upper bounds are the crop counts observed before this operation:

| Split | Crop count |
| --- | ---: |
| train | 30,000 |
| test | 4,328 |
| dev | 2,000 |
| rl | 6,000 |

`reserve` is a source/removal pool and has no preservation target.

## Crop Identity

Match crops using the source frame identity plus a stable crop identifier. Use
explicit source position or detection identity when present, otherwise use crop
index and bbox information. Ambiguous matches are errors and prevent publish.

Review aliases are normalized before selection:

- `approved` and `corrected` become `reviewed`.
- `needs_revision` becomes `uncertain`.
- Legacy review data is used only when it contains an update timestamp.

## Regular Reviewed Data

For every regular-person crop whose normalized status is `reviewed`:

1. Remove all copies from train, test, dev, rl, and reserve.
2. Add exactly one reviewed copy to test under its existing scene-relative
   path.
3. Add `"review_status": "reviewed"` to that crop.

For regular-person crops whose normalized status is `rejected` or `duplicate`,
delete a matching crop when it exists in test. Their copies in other splits are
not changed by the review-status rule. Other statuses do not cause movement or
deletion.

Only reviewed crops receive `review_status`; all other crops must not contain
that field.

## Xiaohongshu Reviewed Data

Only Xiaohongshu crops with normalized status `reviewed` are imported. Group
them by `image.source_id`, falling back to the stable video identity encoded in
the record path when necessary.

Caption similarity uses normalized English text and TF-IDF word 1-2 grams.
Cosine similarity greater than or equal to 0.90 is considered high similarity.
Video groups connected by high-similarity reviewed captions form one allocation
component. An allocation component cannot be split across train and test.

Allocate complete components so that reviewed Xiaohongshu crop counts are as
close as possible to one half in train and one half in test. Components that
conflict with a regular reviewed test caption are pinned to test. Existing
non-reviewed test crops may be removed instead of discarding a reviewed crop.

Every imported Xiaohongshu crop receives `"review_status": "reviewed"`.

## Leakage Prevention And Size Reduction

After finalizing test, compare every train caption with test captions using the
same TF-IDF cosine threshold of 0.90. Remove matching non-reviewed train crops.
Reviewed crops are never discarded. Any reviewed allocation conflict must be
resolved by moving the complete Xiaohongshu allocation component to test.

If train or test exceeds its baseline crop count, reduce it to the baseline by
removing non-reviewed crops. Prefer crops that duplicate another caption, then
prefer lower-confidence and smaller crops. Do not remove reviewed crops to hit a
size target.

Dev and rl receive no replacement samples. They may shrink when regular
reviewed crops move to test. RL must remain a subset of train after removals.

For each train, test, dev, and rl split, a final shortfall of at most 4 percent
of its baseline is accepted. If any shortfall exceeds 4 percent, stop before
publication and report the exact deficit for user guidance.

## Materialization And Publication

Build new train, test, dev, rl, and reserve trees in a staging directory from
the current dataset. Preserve scene-relative paths and the Xiaohongshu subtree.
When multiple retained crops share a frame, write one frame JSON containing
only those crops and update `person_crop_count`. Remove empty frame JSON files.

Do not modify `data/review`. Replace the active split directories only after
all validations succeed. Keep the original directories available during the
swap so a failed publication can restore them, then remove temporary swap data
after success.

Regenerate affected split manifests and statistics so they describe the
published files.

## Validation

Before publication, verify:

1. Every output JSON parses and its `person_crop_count` matches `len(crops)`.
2. Every selected reviewed crop appears exactly once in its assigned split and
   contains `"review_status": "reviewed"`.
3. No non-reviewed crop contains `review_status`.
4. No regular rejected or duplicate reviewed crop remains in test.
5. No train/test caption pair has TF-IDF cosine similarity at least 0.90.
6. No Xiaohongshu video or similarity component crosses train and test.
7. RL remains a crop-level subset of train.
8. No split exceeds its baseline crop count.
9. Any split shortfall is no more than 4 percent; otherwise publication is
   aborted.
10. Manifests and statistics match the materialized split contents.

