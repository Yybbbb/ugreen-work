# Reviewed Prepared Data And RL Length Reward Design

## Goal

Rebuild the prepared Florence person datasets from the reviewed dataset version
published on 2026-08-06, then align the SCST length reward with the desired
18-24 word output range.

## Data Source Of Truth

The authoritative inputs are:

- `data/manifests/{train,dev,test,rl}.jsonl`
- `data/{train,dev,test,rl}/**/*.json`

The rebuilt outputs are:

- `data/prepared/train.jsonl`: 30,000 rows
- `data/prepared/dev.jsonl`: 2,000 rows
- `data/prepared/test.jsonl`: 4,328 rows
- `data/prepared/rl.jsonl`: 5,981 rows
- `data/prepared/metadata.json`: hashes and counts for the rebuilt files and
  their source manifests

The 5,981-row RL count is intentional. It is the reviewed publication's
authoritative manifest count and remains a crop-level subset of reviewed train.
The rebuild must not synthesize or top up 19 samples merely to preserve the old
6,000-row constant.

`data/prepared/native_replay.jsonl` is not rebuilt. It is a separately generated
SFT artifact and is not used by RL.

## Atomic Rebuild

The preparation script must generate all requested splits and metadata in a
new staging directory. It must validate row counts, unique sample IDs, split and
task values, prompts, source captions, attributes, and manifest hashes before
publishing anything.

After every split passes, publish the four JSONL files and metadata as one
logical operation using same-filesystem atomic replacements. If generation or
validation fails, the active `data/prepared` files remain unchanged. Existing
prepared files are preserved until the staged set is complete.

After publication, independently verify:

- prepared row counts equal current manifest row counts;
- every prepared sample ID equals the corresponding manifest sample ID;
- RL sample IDs are a subset of train sample IDs;
- metadata hashes equal the published files and current manifests;
- the reviewed train/test replacements are present in prepared data.

## Unknown Attribute Policy

For RL reward, `unknown`, `none`, `no`, null-like values, and other values
normalized to missing mean the attribute does not exist. Therefore:

- GT unknown + generated unknown is neutral;
- GT unknown + generated concrete value is fabrication;
- GT concrete + generated unknown is a miss;
- GT concrete + generated concrete is match, partial, or conflict.

This policy is authoritative for this experiment and replaces contradictory
text that said unknown could not be negative evidence.

## Length Reward

Replace the non-negative `length_penalty` with a signed `length_score`:

| Caption word count | Length score | Reward contribution at weight 0.10 |
|---|---:|---:|
| 18-24 | +1.0 | +0.10 |
| 12-17 or 25-28 | -0.5 | -0.05 |
| below 12 or above 28 | -1.0 | -0.10 |

The total reward uses `+ W_LENGTH * length_score`. All other reward terms and
weights remain unchanged. Empty or invalid captions still receive `-1.0`
before component scoring.

## Training Contract Consequences

The RL trainer must derive formal sample count and optimizer steps from the
published RL file rather than require exactly 6,000 rows. With 5,981 samples and
global batch 8, one pass is `ceil(5981 / 8) = 748` optimizer steps.

V4A remains the initialization checkpoint. Its existing metrics remain labeled
as results on the old 2026-07-24 prepared test. Before RL model comparison, V4A
must be evaluated once on the rebuilt reviewed test so SFT and RL share the same
test version.

## Testing And Verification

Use the Florence Python 3.10 environment. Tests must cover:

- reviewed full-data expected counts, including RL=5,981;
- staged rebuild failure leaving active prepared files unchanged;
- staged publication updating all requested files and metadata;
- exact signed length scores at every boundary: 11/12/17/18/24/25/28/29;
- reward contribution differences for ideal, near-range, and far-range captions;
- formal RL step calculation returning 748 for 5,981 rows at global batch 8;
- current unknown-to-fabrication behavior remains unchanged.

No RL training or reward service is started as part of this change.
