# Review Caption Attribute Alignment Design

## Goal

Add an in-place, resumable script that aligns every crop caption under
`data/review` with that crop's authoritative structured attributes by calling
the text-only `vllm-qwen36-caption-text` service.

## Attribute Policy

The script removes the top-level `extra` field before prompting. It also drops
empty values and the case-insensitive placeholders `unknown`, `other`, `none`,
`no`, `null`, `n/a`, `na`, `not applicable`, `not_applicable`, and
`unspecified`, including placeholder-only lists and dictionaries. Remaining
values are authoritative: a conflicting caption value is corrected and a
missing meaningful value is naturally added. Unrelated caption facts and
wording remain substantially unchanged. Output must not contain negative
statements about absent, unknown, or unobserved properties.

## Service Protocol

The default endpoint is `http://127.0.0.1:6097/v1/chat/completions`, using the
served model `Qwen36-35b-caption`. Requests are text-only, disable thinking,
and send batches of tasks containing stable IDs, original captions, and
filtered attributes. Responses are JSON objects containing an ID-addressed
caption for every task. Low-temperature decoding favors conservative edits.

## Persistence And In-Place Publication

Validated results are stored as atomic per-crop sidecars under
`data/.review_caption_rewrite_work`. The cache includes the original caption
hash, meaningful-attributes hash, prompt version, model, and rewritten caption.
It supports restart after interruption and prevents an already-applied caption
from being rewritten again.

Successful results are grouped by source JSON. Each JSON is reread immediately
before modification, and a caption is replaced only if it still matches the
task's original caption or already equals the cached result. The complete JSON
is written to a temporary file in the same directory and atomically replaces
the original. No dataset backup is created. Failed or invalid tasks retain
their current caption, are reported, and cause a nonzero exit status.

## Validation And Testing

Responses must contain exactly the requested IDs, nonempty English captions,
no Markdown, plausible length, and no negative phrasing. Tests cover attribute
filtering, prompt construction, response mapping, cache reuse, atomic in-place
updates, conflict detection, retries, and failure preservation. A live smoke
test uses `--limit-crops` so service integration can be checked without
processing the complete dataset.
