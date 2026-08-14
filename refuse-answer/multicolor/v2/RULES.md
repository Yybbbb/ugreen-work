# V2 Pure-Color Gate Rules

## Goal

V2 is a conservative gate before the CNN color classifier.

The detector does not try to prove the visual category "multicolor" exhaustively. It proves whether a masked garment region is pure enough for single-color CNN classification. If the region is not confidently pure, it is treated as multicolor / refuse-risk and should not be sent to the single-color CNN path.

## Inputs

- `image_rgb`: RGB image, `uint8`, shape `(H, W, 3)`.
- `mask`: segmentation mask, shape `(H, W)`.
- `region_id`: target region id, for example `1=upper`, `2=lower`.

If the target ROI has fewer than 50 pixels, the region is not confidently pure.

## Color Space

Use Lab statistics, not RGB-only statistics.

- Use `a*b*` for chromatic color separation.
- Use a separate `L` branch for neutral regions, because black, white, and gray are close in `a*b*`.
- Do not use spatial continuity as a hard filter in V2 pure/multicolor judgment.

Definitions:

```text
C = sqrt(a*a + b*b)
```

`C` is chroma. Low-chroma pixels are black / white / gray candidates.

## Cluster Ratios

After candidate color centers are found, assign every valid ROI pixel to the nearest merged color cluster and compute ratios from pixel area, not from histogram peak height.

Required metrics:

```text
main_ratio          = largest cluster area / ROI area
second_ratio        = second largest cluster area / ROI area, or 0 if absent
minor_total_ratio   = 1 - main_ratio
n_effective_colors  = number of clusters whose area ratio >= EFFECTIVE_CLUSTER_MIN_RATIO
```

Recommended initial constants:

```text
MIN_ROI_PIXELS = 50
EFFECTIVE_CLUSTER_MIN_RATIO = 0.02
MAIN_RATIO_PURE_MIN = 0.80
SECOND_RATIO_PURE_MAX = 0.10
MINOR_TOTAL_PURE_MAX = 0.20
MULTI_SMALL_COLORS_MIN_COUNT = 3
MULTI_SMALL_COLORS_MINOR_TOTAL_MIN = 0.15
```

Use strict `<` for pure-side upper bounds so the `0.20` boundary is not ambiguous.

## Neutral L Branch

The neutral branch is used to catch black / white / gray multicolor patterns that are invisible in `a*b*`.

Recommended initial constants:

```text
NEUTRAL_CHROMA_MAX = 12.0
NEUTRAL_REGION_RATIO_MIN = 0.60
L_PEAK_MIN_RATIO = 0.10
L_PEAK_MIN_DELTA = 30.0
L_MINOR_TOTAL_MIN = 0.20
```

Set `neutral_l_multipeak = true` when:

```text
neutral_pixel_ratio >= NEUTRAL_REGION_RATIO_MIN
and at least two L peaks exist
and the strongest two L peaks differ by >= L_PEAK_MIN_DELTA
and (
    second_l_peak_ratio >= L_PEAK_MIN_RATIO
    or l_minor_total_ratio >= L_MINOR_TOTAL_MIN
)
```

This catches black-white stripes, black-white blocks, gray-white blocks, and similar neutral multicolor regions.

For chromatic regions, ordinary `L` variation should not split a pure garment. A red shirt with shadows should remain pure if its `a*b*` cluster is dominant.

## Final Decision

V2 uses "pure first, everything else reject-risk" logic.

```text
pure =
    main_ratio >= MAIN_RATIO_PURE_MIN
    and second_ratio < SECOND_RATIO_PURE_MAX
    and minor_total_ratio < MINOR_TOTAL_PURE_MAX
    and not (
        n_effective_colors >= MULTI_SMALL_COLORS_MIN_COUNT
        and minor_total_ratio >= MULTI_SMALL_COLORS_MINOR_TOTAL_MIN
    )
    and not neutral_l_multipeak

is_multicolor = not pure
```

There is no outside set in the final binary decision:

```text
if pure:
    send to CNN color classifier
else:
    treat as multicolor / refuse-risk
```

## Examples

```text
main=0.86, second=0.07, minor_total=0.14, n_effective=2, neutral_l=false
=> pure

main=0.82, second=0.09, minor_total=0.18, n_effective=3, neutral_l=false
=> multicolor / refuse-risk

main=0.76, second=0.08, minor_total=0.24, n_effective=4, neutral_l=false
=> multicolor / refuse-risk

main=0.91, second=0.05, minor_total=0.09, n_effective=2, neutral_l=true
=> multicolor / refuse-risk

main=0.80, second=0.09, minor_total=0.20, n_effective=2, neutral_l=false
=> multicolor / refuse-risk because pure requires minor_total < 0.20
```

## Reporting Fields

Each region result should include:

```json
{
  "region": "upper",
  "is_pure": false,
  "is_multicolor": true,
  "main_ratio": 0.76,
  "second_ratio": 0.08,
  "minor_total_ratio": 0.24,
  "n_effective_colors": 4,
  "neutral_l_multipeak": false,
  "decision_reason": "minor_total_ratio>=0.20",
  "clusters": [
    {"lab": [42.0, 61.0, 35.0], "ratio": 0.76},
    {"lab": [68.0, -28.0, 46.0], "ratio": 0.08}
  ]
}
```

`decision_reason` should report the first pure-failure reason in the final decision order.
