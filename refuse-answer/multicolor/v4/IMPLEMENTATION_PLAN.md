# V4 Shadow Merge Experiment Plan

Goal: reduce pure-color false rejects caused by highlights, shadows, and folds being split into separate Lab clusters.

Architecture:
- Reuse V3 adaptive Lab clustering as the first stage.
- Add a post-clustering `shadow_merge` stage that merges non-neutral clusters with similar `a*b*` hue/chroma but different `L`.
- Keep neutral clusters separate so black, white, and gray patterns remain detectable.
- Keep V3 purity thresholds unchanged, then compare metrics against V3.

Validation:
- Unit tests must prove chromatic shadow clusters merge.
- Unit tests must prove black/white neutral clusters do not merge.
- Detector tests must prove a same-color light/dark region is pure and a black/white region is multicolor.
- Full evaluation runs on `val_pure` and `val_multicolor/upper_multicolor`.
