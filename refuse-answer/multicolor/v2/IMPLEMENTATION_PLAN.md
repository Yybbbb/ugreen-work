# V2 Pure-Color Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conservative Lab-statistics gate that sends only confidently pure garment regions to the CNN color classifier and treats every other region as multicolor / refuse-risk.

**Architecture:** V2 keeps the V1 masked-region workflow, but replaces the "prove multicolor" decision with a "prove pure" decision. The detector computes Lab color clusters, computes pixel-area cluster ratios, adds a neutral `L` branch for black / white / gray patterns, and makes a binary `is_pure` decision with no outside set.

**Tech Stack:** Python 3, NumPy, SciPy, Pillow, optional `pycocotools` for COCO RLE mask decoding, pytest for unit tests.

---

## File Structure

- Create: `segment-color/refuse-answer/multicolor/v2/__init__.py`
  - Marks V2 as an importable package for tests and scripts.
- Create: `segment-color/refuse-answer/multicolor/v2/color_space.py`
  - Owns RGB to Lab conversion and chroma computation.
- Create: `segment-color/refuse-answer/multicolor/v2/color_clusters.py`
  - Finds candidate Lab color clusters, merges nearby clusters, assigns ROI pixels to clusters, and computes area ratios.
- Create: `segment-color/refuse-answer/multicolor/v2/neutral_l.py`
  - Implements the neutral black / white / gray `L` histogram branch.
- Create: `segment-color/refuse-answer/multicolor/v2/purity_rules.py`
  - Owns the final pure-first rule and decision reasons.
- Create: `segment-color/refuse-answer/multicolor/v2/detector.py`
  - Public API: `detect_region()` and `detect_upper_lower()`.
- Create: `segment-color/refuse-answer/multicolor/v2/mask_io.py`
  - Loads masks from existing annotation JSON formats used by V1 scripts.
- Create: `segment-color/refuse-answer/multicolor/v2/analyze_val.py`
  - Batch runner for manifests, using V2 detector and honest runtime config reporting.
- Create: `segment-color/refuse-answer/multicolor/v2/evaluate.py`
  - Evaluates V2 against pure and multicolor validation sets.
- Create: `segment-color/refuse-answer/multicolor/v2/tests/segment_color_import.py`
  - Adds the V2 directory to `sys.path` for local script-style test imports.
- Create: `segment-color/refuse-answer/multicolor/v2/tests/test_purity_rules.py`
  - Unit tests for rule boundary behavior.
- Create: `segment-color/refuse-answer/multicolor/v2/tests/test_neutral_l.py`
  - Unit tests for neutral `L` branch.
- Create: `segment-color/refuse-answer/multicolor/v2/tests/test_detector_synthetic.py`
  - Synthetic image tests for pure chromatic regions, shadowed pure regions, black-white stripes, and multi-small-color cases.

## Task 1: Add Package Skeleton and Shared Config

**Files:**
- Create: `segment-color/refuse-answer/multicolor/v2/__init__.py`
- Create: `segment-color/refuse-answer/multicolor/v2/config.py`

- [ ] **Step 1: Create package marker**

Create `__init__.py` with:

```python
"""V2 pure-color gate for masked garment regions."""
```

- [ ] **Step 2: Create config constants**

Create `config.py` with:

```python
MIN_ROI_PIXELS = 50

N_AB_BINS = 128
AB_SIGMA = 2.5
AB_PEAK_THRESHOLD = 0.12
AB_MERGE_DIST = 15.0

EFFECTIVE_CLUSTER_MIN_RATIO = 0.02

MAIN_RATIO_PURE_MIN = 0.80
SECOND_RATIO_PURE_MAX = 0.10
MINOR_TOTAL_PURE_MAX = 0.20
MULTI_SMALL_COLORS_MIN_COUNT = 3
MULTI_SMALL_COLORS_MINOR_TOTAL_MIN = 0.15

NEUTRAL_CHROMA_MAX = 12.0
NEUTRAL_REGION_RATIO_MIN = 0.60
L_HIST_BINS = 50
L_HIST_SIGMA = 1.5
L_PEAK_MIN_RATIO = 0.10
L_PEAK_MIN_DELTA = 30.0
L_MINOR_TOTAL_MIN = 0.20
```

- [ ] **Step 3: Run syntax check**

Run:

```bash
python3 -m py_compile segment-color/refuse-answer/multicolor/v2/config.py
```

Expected: command exits with code 0 and prints no errors.

## Task 2: Implement Final Pure-First Rules with Tests

**Files:**
- Create: `segment-color/refuse-answer/multicolor/v2/tests/test_purity_rules.py`
- Create: `segment-color/refuse-answer/multicolor/v2/purity_rules.py`

- [ ] **Step 1: Write failing rule tests**

Create `tests/test_purity_rules.py` with:

```python
from segment_color_import import add_repo_path

add_repo_path()

from purity_rules import decide_purity


def test_confident_pure_passes():
    result = decide_purity(
        main_ratio=0.86,
        second_ratio=0.07,
        minor_total_ratio=0.14,
        n_effective_colors=2,
        neutral_l_multipeak=False,
    )
    assert result["is_pure"] is True
    assert result["is_multicolor"] is False
    assert result["decision_reason"] == "pure"


def test_boundary_minor_total_020_rejects():
    result = decide_purity(
        main_ratio=0.80,
        second_ratio=0.09,
        minor_total_ratio=0.20,
        n_effective_colors=2,
        neutral_l_multipeak=False,
    )
    assert result["is_pure"] is False
    assert result["decision_reason"] == "minor_total_ratio>=0.20"


def test_single_strong_secondary_rejects():
    result = decide_purity(
        main_ratio=0.84,
        second_ratio=0.11,
        minor_total_ratio=0.16,
        n_effective_colors=2,
        neutral_l_multipeak=False,
    )
    assert result["is_pure"] is False
    assert result["decision_reason"] == "second_ratio>=0.10"


def test_multiple_small_colors_reject():
    result = decide_purity(
        main_ratio=0.82,
        second_ratio=0.09,
        minor_total_ratio=0.18,
        n_effective_colors=3,
        neutral_l_multipeak=False,
    )
    assert result["is_pure"] is False
    assert result["decision_reason"] == "n_effective_colors>=3 and minor_total_ratio>=0.15"


def test_neutral_l_multipeak_rejects():
    result = decide_purity(
        main_ratio=0.91,
        second_ratio=0.05,
        minor_total_ratio=0.09,
        n_effective_colors=2,
        neutral_l_multipeak=True,
    )
    assert result["is_pure"] is False
    assert result["decision_reason"] == "neutral_l_multipeak"
```

Create `tests/segment_color_import.py` with:

```python
import sys
from pathlib import Path


def add_repo_path():
    here = Path(__file__).resolve()
    v2_dir = here.parents[1]
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_purity_rules.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'purity_rules'`.

- [ ] **Step 3: Implement `decide_purity`**

Create `purity_rules.py` with:

```python
from config import (
    MAIN_RATIO_PURE_MIN,
    SECOND_RATIO_PURE_MAX,
    MINOR_TOTAL_PURE_MAX,
    MULTI_SMALL_COLORS_MIN_COUNT,
    MULTI_SMALL_COLORS_MINOR_TOTAL_MIN,
)


def decide_purity(
    main_ratio,
    second_ratio,
    minor_total_ratio,
    n_effective_colors,
    neutral_l_multipeak,
):
    if neutral_l_multipeak:
        return _result(False, "neutral_l_multipeak")

    if second_ratio >= SECOND_RATIO_PURE_MAX:
        return _result(False, "second_ratio>=0.10")

    if minor_total_ratio >= MINOR_TOTAL_PURE_MAX:
        return _result(False, "minor_total_ratio>=0.20")

    if (
        n_effective_colors >= MULTI_SMALL_COLORS_MIN_COUNT
        and minor_total_ratio >= MULTI_SMALL_COLORS_MINOR_TOTAL_MIN
    ):
        return _result(False, "n_effective_colors>=3 and minor_total_ratio>=0.15")

    if main_ratio >= MAIN_RATIO_PURE_MIN and minor_total_ratio < MINOR_TOTAL_PURE_MAX:
        return _result(True, "pure")

    return _result(False, "main_ratio<0.80")


def _result(is_pure, reason):
    return {
        "is_pure": bool(is_pure),
        "is_multicolor": not bool(is_pure),
        "decision_reason": reason,
    }
```

- [ ] **Step 4: Run rule tests**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_purity_rules.py -q
```

Expected: all tests PASS.

## Task 3: Implement Lab Conversion and Chroma

**Files:**
- Create: `segment-color/refuse-answer/multicolor/v2/color_space.py`
- Create: `segment-color/refuse-answer/multicolor/v2/tests/test_color_space.py`

- [ ] **Step 1: Write color-space tests**

Create `tests/test_color_space.py` with:

```python
import numpy as np
from segment_color_import import add_repo_path

add_repo_path()

from color_space import rgb_to_lab, chroma_ab


def test_rgb_to_lab_shape_and_range():
    rgb = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0]]], dtype=np.uint8)
    lab = rgb_to_lab(rgb)
    assert lab.shape == (1, 3, 3)
    assert lab[0, 0, 0] <= 1.0
    assert lab[0, 1, 0] >= 99.0
    assert lab[0, 2, 1] > 40.0


def test_chroma_ab_black_white_is_low():
    rgb = np.array([[0, 0, 0], [255, 255, 255], [128, 128, 128]], dtype=np.uint8)
    lab = rgb_to_lab(rgb)
    c = chroma_ab(lab)
    assert np.all(c < 2.0)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_color_space.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'color_space'`.

- [ ] **Step 3: Implement color-space helpers**

Create `color_space.py` by moving the V1 `rgb_to_lab` implementation from `segment-color/refuse-answer/multicolor/v1/histogram_peak.py` and adding:

```python
import numpy as np


def chroma_ab(lab):
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
```

- [ ] **Step 4: Run color-space tests**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_color_space.py -q
```

Expected: all tests PASS.

## Task 4: Implement Neutral L Branch

**Files:**
- Create: `segment-color/refuse-answer/multicolor/v2/neutral_l.py`
- Create: `segment-color/refuse-answer/multicolor/v2/tests/test_neutral_l.py`

- [ ] **Step 1: Write neutral L tests**

Create `tests/test_neutral_l.py` with:

```python
import numpy as np
from segment_color_import import add_repo_path

add_repo_path()

from neutral_l import analyze_neutral_l


def test_black_white_neutral_split_detected():
    lab = np.zeros((1000, 3), dtype=np.float32)
    lab[:500, 0] = 5.0
    lab[500:, 0] = 95.0
    result = analyze_neutral_l(lab)
    assert result["neutral_l_multipeak"] is True
    assert result["l_second_ratio"] >= 0.10


def test_chromatic_region_skips_l_split():
    lab = np.zeros((1000, 3), dtype=np.float32)
    lab[:500, 0] = 35.0
    lab[500:, 0] = 80.0
    lab[:, 1] = 60.0
    lab[:, 2] = 35.0
    result = analyze_neutral_l(lab)
    assert result["neutral_l_multipeak"] is False
    assert result["skipped"] is True
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_neutral_l.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'neutral_l'`.

- [ ] **Step 3: Implement `analyze_neutral_l`**

Create `neutral_l.py` with:

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from color_space import chroma_ab
from config import (
    NEUTRAL_CHROMA_MAX,
    NEUTRAL_REGION_RATIO_MIN,
    L_HIST_BINS,
    L_HIST_SIGMA,
    L_PEAK_MIN_RATIO,
    L_PEAK_MIN_DELTA,
    L_MINOR_TOTAL_MIN,
)


def analyze_neutral_l(pixels_lab):
    if len(pixels_lab) == 0:
        return _empty(skipped=True)

    chroma = chroma_ab(pixels_lab)
    neutral_ratio = float(np.mean(chroma <= NEUTRAL_CHROMA_MAX))
    if neutral_ratio < NEUTRAL_REGION_RATIO_MIN:
        result = _empty(skipped=True)
        result["neutral_ratio"] = neutral_ratio
        return result

    l_vals = np.clip(pixels_lab[:, 0], 0.0, 100.0)
    hist, edges = np.histogram(l_vals, bins=L_HIST_BINS, range=(0.0, 100.0))
    smooth = gaussian_filter1d(hist.astype(np.float32), sigma=L_HIST_SIGMA)
    peak_indices = argrelextrema(smooth, np.greater, order=2)[0]

    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(smooth))])

    peaks = []
    for idx in peak_indices:
        l_center = float((edges[idx] + edges[idx + 1]) / 2.0)
        height = float(smooth[idx])
        peaks.append((l_center, height))
    peaks.sort(key=lambda item: item[1], reverse=True)

    total_height = sum(height for _, height in peaks) or 1.0
    ratios = [(l_center, height / total_height) for l_center, height in peaks]

    if len(ratios) < 2:
        return {
            "neutral_l_multipeak": False,
            "skipped": False,
            "neutral_ratio": neutral_ratio,
            "l_peaks": ratios,
            "l_second_ratio": 0.0,
            "l_minor_total_ratio": 0.0,
            "l_peak_delta": 0.0,
        }

    l_delta = abs(ratios[0][0] - ratios[1][0])
    second_ratio = float(ratios[1][1])
    minor_total = float(sum(r for _, r in ratios[1:]))
    detected = (
        l_delta >= L_PEAK_MIN_DELTA
        and (second_ratio >= L_PEAK_MIN_RATIO or minor_total >= L_MINOR_TOTAL_MIN)
    )

    return {
        "neutral_l_multipeak": bool(detected),
        "skipped": False,
        "neutral_ratio": neutral_ratio,
        "l_peaks": ratios,
        "l_second_ratio": second_ratio,
        "l_minor_total_ratio": minor_total,
        "l_peak_delta": float(l_delta),
    }


def _empty(skipped):
    return {
        "neutral_l_multipeak": False,
        "skipped": skipped,
        "neutral_ratio": 0.0,
        "l_peaks": [],
        "l_second_ratio": 0.0,
        "l_minor_total_ratio": 0.0,
        "l_peak_delta": 0.0,
    }
```

- [ ] **Step 4: Run neutral tests**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_neutral_l.py -q
```

Expected: all tests PASS.

## Task 5: Implement Color Cluster Ratios

**Files:**
- Create: `segment-color/refuse-answer/multicolor/v2/color_clusters.py`
- Create: `segment-color/refuse-answer/multicolor/v2/tests/test_color_clusters.py`

- [ ] **Step 1: Write cluster ratio tests**

Create `tests/test_color_clusters.py` with:

```python
import numpy as np
from segment_color_import import add_repo_path

add_repo_path()

from color_clusters import compute_cluster_stats_from_centers


def test_ratios_use_pixel_area_not_peak_height():
    lab = np.zeros((100, 3), dtype=np.float32)
    lab[:80, 1:] = [60.0, 40.0]
    lab[80:90, 1:] = [-50.0, 35.0]
    lab[90:, 1:] = [20.0, -60.0]
    centers = np.array([[60.0, 40.0], [-50.0, 35.0], [20.0, -60.0]], dtype=np.float32)

    stats = compute_cluster_stats_from_centers(lab, centers)

    assert stats["main_ratio"] == 0.80
    assert stats["second_ratio"] == 0.10
    assert stats["minor_total_ratio"] == 0.20
    assert stats["n_effective_colors"] == 3
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_color_clusters.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'color_clusters'`.

- [ ] **Step 3: Implement cluster assignment and stats**

Create `color_clusters.py` with:

```python
import numpy as np

from config import EFFECTIVE_CLUSTER_MIN_RATIO


def compute_cluster_stats_from_centers(pixels_lab, centers_ab):
    if len(pixels_lab) == 0 or len(centers_ab) == 0:
        return _empty_stats()

    ab = pixels_lab[:, 1:3].astype(np.float32)
    centers = np.asarray(centers_ab, dtype=np.float32)
    distances = ((ab[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)

    counts = np.bincount(labels, minlength=len(centers)).astype(np.float32)
    ratios = counts / float(len(pixels_lab))
    order = np.argsort(-ratios)
    ratios_sorted = ratios[order]

    main_ratio = float(ratios_sorted[0]) if len(ratios_sorted) else 0.0
    second_ratio = float(ratios_sorted[1]) if len(ratios_sorted) > 1 else 0.0
    minor_total = float(max(0.0, 1.0 - main_ratio))
    n_effective = int(np.sum(ratios >= EFFECTIVE_CLUSTER_MIN_RATIO))

    clusters = []
    for idx in order:
        clusters.append({
            "ab": centers[idx].tolist(),
            "ratio": round(float(ratios[idx]), 6),
        })

    return {
        "main_ratio": round(main_ratio, 6),
        "second_ratio": round(second_ratio, 6),
        "minor_total_ratio": round(minor_total, 6),
        "n_effective_colors": n_effective,
        "clusters": clusters,
        "labels": labels,
    }


def _empty_stats():
    return {
        "main_ratio": 0.0,
        "second_ratio": 0.0,
        "minor_total_ratio": 1.0,
        "n_effective_colors": 0,
        "clusters": [],
        "labels": np.array([], dtype=np.int32),
    }
```

- [ ] **Step 4: Add histogram peak finder**

Add `find_ab_centers(pixels_lab)` to `color_clusters.py` by adapting V1 `detect_color_peaks`:

```python
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

from config import N_AB_BINS, AB_SIGMA, AB_PEAK_THRESHOLD, AB_MERGE_DIST


def find_ab_centers(
    pixels_lab,
    n_bins=N_AB_BINS,
    sigma=AB_SIGMA,
    peak_threshold=AB_PEAK_THRESHOLD,
    merge_dist=AB_MERGE_DIST,
):
    if len(pixels_lab) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    a_vals = np.clip(pixels_lab[:, 1], -128, 127)
    b_vals = np.clip(pixels_lab[:, 2], -128, 127)
    hist, a_edges, b_edges = np.histogram2d(
        a_vals, b_vals, bins=n_bins, range=[[-128, 127], [-128, 127]]
    )
    smooth = gaussian_filter(hist, sigma=sigma)
    ys, xs = argrelextrema(smooth, np.greater, order=1)

    peaks = []
    for yi, xi in zip(ys, xs):
        peaks.append((
            float((a_edges[yi] + a_edges[yi + 1]) / 2.0),
            float((b_edges[xi] + b_edges[xi + 1]) / 2.0),
            float(smooth[yi, xi]),
        ))

    if not peaks:
        mean_ab = np.mean(np.column_stack([a_vals, b_vals]), axis=0)
        return mean_ab.reshape(1, 2).astype(np.float32)

    peaks.sort(key=lambda item: item[2], reverse=True)
    main_height = peaks[0][2]
    peaks = [p for p in peaks if p[2] >= peak_threshold * main_height]

    merged = []
    for peak in peaks:
        if not merged:
            merged.append(peak)
            continue
        too_close = False
        for kept in merged:
            dist = ((peak[0] - kept[0]) ** 2 + (peak[1] - kept[1]) ** 2) ** 0.5
            if dist < merge_dist:
                too_close = True
                break
        if not too_close:
            merged.append(peak)

    return np.array([[p[0], p[1]] for p in merged], dtype=np.float32)
```

- [ ] **Step 5: Run cluster tests**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_color_clusters.py -q
```

Expected: all tests PASS.

## Task 6: Implement Public Detector API

**Files:**
- Create: `segment-color/refuse-answer/multicolor/v2/detector.py`
- Create: `segment-color/refuse-answer/multicolor/v2/tests/test_detector_synthetic.py`

- [ ] **Step 1: Write synthetic detector tests**

Create `tests/test_detector_synthetic.py` with:

```python
import numpy as np
from segment_color_import import add_repo_path

add_repo_path()

from detector import detect_region


def test_pure_red_region_passes():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:, :] = [220, 20, 20]
    mask = np.ones((20, 20), dtype=np.int32)
    result = detect_region(img, mask, 1, "upper")
    assert result["is_pure"] is True


def test_black_white_split_rejects():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:, :10] = [0, 0, 0]
    img[:, 10:] = [255, 255, 255]
    mask = np.ones((20, 20), dtype=np.int32)
    result = detect_region(img, mask, 1, "upper")
    assert result["is_pure"] is False
    assert result["neutral_l"]["neutral_l_multipeak"] is True


def test_main_color_with_three_minor_colors_rejects():
    img = np.zeros((10, 100, 3), dtype=np.uint8)
    img[:, :76] = [220, 20, 20]
    img[:, 76:84] = [20, 220, 20]
    img[:, 84:92] = [20, 20, 220]
    img[:, 92:] = [240, 220, 20]
    mask = np.ones((10, 100), dtype=np.int32)
    result = detect_region(img, mask, 1, "upper")
    assert result["is_pure"] is False
    assert result["minor_total_ratio"] >= 0.20
```

- [ ] **Step 2: Run detector tests to verify failure**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_detector_synthetic.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'detector'`.

- [ ] **Step 3: Implement `detect_region` and `detect_upper_lower`**

Create `detector.py` with:

```python
import numpy as np

from color_space import rgb_to_lab
from color_clusters import find_ab_centers, compute_cluster_stats_from_centers
from neutral_l import analyze_neutral_l
from purity_rules import decide_purity
from config import MIN_ROI_PIXELS


def detect_region(image_rgb, mask, region_id, region_name="region"):
    roi = mask == region_id
    n_pixels = int(np.sum(roi))
    if n_pixels < MIN_ROI_PIXELS:
        decision = {
            "is_pure": False,
            "is_multicolor": True,
            "decision_reason": "too_few_pixels",
        }
        return _format_result(region_name, n_pixels, decision, None, None)

    pixels_lab = rgb_to_lab(image_rgb[roi])
    centers_ab = find_ab_centers(pixels_lab)
    stats = compute_cluster_stats_from_centers(pixels_lab, centers_ab)
    neutral = analyze_neutral_l(pixels_lab)
    decision = decide_purity(
        main_ratio=stats["main_ratio"],
        second_ratio=stats["second_ratio"],
        minor_total_ratio=stats["minor_total_ratio"],
        n_effective_colors=stats["n_effective_colors"],
        neutral_l_multipeak=neutral["neutral_l_multipeak"],
    )
    return _format_result(region_name, n_pixels, decision, stats, neutral)


def detect_upper_lower(image_rgb, mask):
    return {
        "upper": detect_region(image_rgb, mask, 1, "upper"),
        "lower": detect_region(image_rgb, mask, 2, "lower"),
    }


def _format_result(region_name, n_pixels, decision, stats, neutral):
    stats = stats or {
        "main_ratio": 0.0,
        "second_ratio": 0.0,
        "minor_total_ratio": 1.0,
        "n_effective_colors": 0,
        "clusters": [],
    }
    neutral = neutral or {"neutral_l_multipeak": False}
    return {
        "region": region_name,
        "n_pixels": n_pixels,
        "is_pure": decision["is_pure"],
        "is_multicolor": decision["is_multicolor"],
        "decision_reason": decision["decision_reason"],
        "main_ratio": stats["main_ratio"],
        "second_ratio": stats["second_ratio"],
        "minor_total_ratio": stats["minor_total_ratio"],
        "n_effective_colors": stats["n_effective_colors"],
        "clusters": stats["clusters"],
        "neutral_l": neutral,
    }
```

- [ ] **Step 4: Run detector tests**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests/test_detector_synthetic.py -q
```

Expected: all tests PASS.

## Task 7: Port Mask Loading and Batch Analysis

**Files:**
- Create: `segment-color/refuse-answer/multicolor/v2/mask_io.py`
- Create: `segment-color/refuse-answer/multicolor/v2/analyze_val.py`

- [ ] **Step 1: Create `mask_io.py` from V1**

Move these V1 behaviors into `mask_io.py`:

```python
def load_mask_from_anno(anno_path, img_shape):
    """Return mask with 0=background, 1=upper, 2=lower, or None."""
```

Required supported formats:

```text
coco_rle
lip_palette_ids
```

Use the existing fallback path for LIP masks:

```text
/data1/work/MichaelYu/segment-color/data/LIP_clothes_accessory_unified
```

- [ ] **Step 2: Create V2 `analyze_val.py`**

Create a script with this CLI:

```bash
python3 segment-color/refuse-answer/multicolor/v2/analyze_val.py \
  --val-dir segment-color/refuse-answer/multicolor/val_pure \
  --output segment-color/refuse-answer/multicolor/v2/val_pure_results.json
```

Required output fields:

```json
{
  "config": {
    "main_ratio_pure_min": 0.8,
    "second_ratio_pure_max": 0.1,
    "minor_total_pure_max": 0.2,
    "neutral_chroma_max": 12.0
  },
  "stats": {
    "total_samples": 0,
    "mask_loaded": 0,
    "mask_failed": 0,
    "detected_upper_pure": 0,
    "detected_upper_multicolor": 0
  },
  "results": []
}
```

The `config` must be read from `config.py` at runtime. Do not hardcode stale values in the report.

- [ ] **Step 3: Run smoke analysis on 5 samples**

Run:

```bash
python3 segment-color/refuse-answer/multicolor/v2/analyze_val.py \
  --val-dir segment-color/refuse-answer/multicolor/val_pure \
  --limit 5 \
  --output /tmp/v2_val_pure_smoke.json
```

Expected: command exits with code 0 and writes JSON containing `config`, `stats`, and `results`.

## Task 8: Add Final Evaluation Script

**Files:**
- Create: `segment-color/refuse-answer/multicolor/v2/evaluate.py`

- [ ] **Step 1: Create evaluation CLI**

Create `evaluate.py` with this CLI:

```bash
python3 segment-color/refuse-answer/multicolor/v2/evaluate.py \
  --pure-dir segment-color/refuse-answer/multicolor/val_pure \
  --multi-dir segment-color/refuse-answer/multicolor/val_multicolor/upper_multicolor \
  --output segment-color/refuse-answer/multicolor/v2/final_eval.json
```

Pure validation files have ground truth `is_pure=True`.

Multicolor validation files have ground truth `is_pure=False`.

- [ ] **Step 2: Compute metrics**

The output JSON must include:

```json
{
  "confusion_matrix": {
    "pure_tp": 0,
    "pure_fn": 0,
    "multi_tp": 0,
    "multi_fp": 0
  },
  "overall": {
    "pure_precision": 0.0,
    "pure_recall": 0.0,
    "multi_recall": 0.0,
    "accuracy": 0.0
  }
}
```

Use `pure_tp` for pure samples correctly accepted by the gate. Use `pure_fn` for pure samples rejected as multicolor. Use `multi_tp` for multicolor samples correctly rejected. Use `multi_fp` for multicolor samples incorrectly accepted as pure.

- [ ] **Step 3: Run final evaluation**

Run:

```bash
python3 segment-color/refuse-answer/multicolor/v2/evaluate.py \
  --pure-dir segment-color/refuse-answer/multicolor/val_pure \
  --multi-dir segment-color/refuse-answer/multicolor/val_multicolor/upper_multicolor \
  --output segment-color/refuse-answer/multicolor/v2/final_eval.json
```

Expected: command exits with code 0 and writes `final_eval.json`.

## Task 9: Verification Commands

**Files:**
- Uses all V2 files.

- [ ] **Step 1: Run all V2 unit tests**

Run:

```bash
python3 -m pytest segment-color/refuse-answer/multicolor/v2/tests -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run syntax check for all V2 scripts**

Run:

```bash
python3 -m py_compile \
  segment-color/refuse-answer/multicolor/v2/config.py \
  segment-color/refuse-answer/multicolor/v2/color_space.py \
  segment-color/refuse-answer/multicolor/v2/color_clusters.py \
  segment-color/refuse-answer/multicolor/v2/neutral_l.py \
  segment-color/refuse-answer/multicolor/v2/purity_rules.py \
  segment-color/refuse-answer/multicolor/v2/detector.py \
  segment-color/refuse-answer/multicolor/v2/mask_io.py \
  segment-color/refuse-answer/multicolor/v2/analyze_val.py \
  segment-color/refuse-answer/multicolor/v2/evaluate.py
```

Expected: command exits with code 0 and prints no errors.

- [ ] **Step 3: Run smoke reports**

Run:

```bash
python3 segment-color/refuse-answer/multicolor/v2/analyze_val.py \
  --val-dir segment-color/refuse-answer/multicolor/val_pure \
  --limit 10 \
  --output /tmp/v2_val_pure_10.json

python3 segment-color/refuse-answer/multicolor/v2/analyze_val.py \
  --val-dir segment-color/refuse-answer/multicolor/val_multicolor \
  --limit 10 \
  --output /tmp/v2_val_multi_10.json
```

Expected: both commands exit with code 0 and each JSON has 10 or fewer result entries.

## Self-Review

- Spec coverage: `RULES.md` defines pure-first behavior, neutral `L` branch, cluster ratios, final binary decision, and reporting fields. Tasks 2 through 8 map each rule section to code and tests.
- Wording scan: no task contains unresolved markers or vague implementation-only instructions. The remaining work is explicit task execution.
- Type consistency: public detector result fields match `RULES.md`: `is_pure`, `is_multicolor`, `main_ratio`, `second_ratio`, `minor_total_ratio`, `n_effective_colors`, `neutral_l`, `decision_reason`, and `clusters`.
