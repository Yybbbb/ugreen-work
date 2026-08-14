#!/usr/bin/env python3
"""
Analyze inference results from train.txt and generate an HTML report.
Usage:
  python analyze_and_report.py --results infer_train_rgb/results.jsonl --output report.html
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

COLOR_NAMES = ["unknown", "black", "white", "gray", "red", "yellow",
               "green", "blue", "purple", "pink", "orange", "brown"]
COLOR_ZH = {"unknown": "未知", "black": "黑", "white": "白", "gray": "灰",
            "red": "红", "yellow": "黄", "green": "绿", "blue": "蓝",
            "purple": "紫", "pink": "粉", "orange": "橙", "brown": "棕"}
SEG_CLASS_NAMES = ["background", "upper", "lower", "dress",
                    "hat", "glasses", "mask", "hair"]


def load_results(path):
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def make_hist(values, bins):
    hist = []
    total = len(values)
    for lo, hi, label in bins:
        cnt = sum(1 for v in values if lo <= v < hi)
        hist.append((label, cnt, cnt / total * 100 if total > 0 else 0))
    return hist


def compute_precision(bucket, garment):
    correct = 0; total_pred = 0
    for r in bucket:
        gt_key = f"{garment}_gt"; pred_key = f"{garment}_pred"
        if r[gt_key] != 0:
            total_pred += 1
            if r[pred_key] == r[gt_key]:
                correct += 1
    return correct / max(1, total_pred)


def compute_overall_precision(subset):
    correct = 0; total = 0
    for r in subset:
        if r["upper_gt"] != 0:
            total += 1
            if r["upper_pred"] == r["upper_gt"]:
                correct += 1
        if r["lower_gt"] != 0:
            total += 1
            if r["lower_pred"] == r["lower_gt"]:
                correct += 1
    return correct / max(1, total)


def top_colors_by_precision(bucket, garment, top_k=3, min_preds=5):
    per_color = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in bucket:
        gt_key = f"{garment}_gt"; pred_key = f"{garment}_pred"
        if r[gt_key] != 0:
            pred_color = r[pred_key]
            per_color[pred_color]["total"] += 1
            if r[pred_key] == r[gt_key]:
                per_color[pred_color]["correct"] += 1
    scores = []
    for color_id, stats in per_color.items():
        if stats["total"] >= min_preds:
            prec = stats["correct"] / stats["total"]
            scores.append({"color": COLOR_NAMES[color_id], "color_zh": COLOR_ZH[COLOR_NAMES[color_id]], "prec": round(prec, 3), "n": stats["total"]})
    scores.sort(key=lambda x: x["prec"], reverse=True)
    return scores[:top_k]


def global_color_precision(results, garment):
    per_color = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        gt_key = f"{garment}_gt"; pred_key = f"{garment}_pred"
        if r[gt_key] != 0:
            pred_color = r[pred_key]
            per_color[pred_color]["total"] += 1
            if r[pred_key] == r[gt_key]:
                per_color[pred_color]["correct"] += 1
    scores = []
    for color_id, stats in per_color.items():
        prec = stats["correct"] / max(1, stats["total"])
        scores.append({"color": COLOR_NAMES[color_id], "color_zh": COLOR_ZH[COLOR_NAMES[color_id]], "prec": round(prec, 3), "n_pred": stats["total"]})
    scores.sort(key=lambda x: x["prec"], reverse=True)
    return scores


def compute_all_stats(results):
    """Compute all statistics from results and return a dict for template rendering."""
    total = len(results)
    widths = [r["orig_W"] for r in results if r["orig_W"] > 0]
    heights = [r["orig_H"] for r in results if r["orig_H"] > 0]
    areas = [r["area_kpx"] for r in results if r["area_kpx"] > 0]
    short_edges = [r["short_edge"] for r in results if r["short_edge"] > 0]
    aspects = [r["aspect_ratio"] for r in results if r["aspect_ratio"] > 0]
    unique_sizes = len(set((r["orig_W"], r["orig_H"]) for r in results))

    stats = {
        "total": total,
        "mean_w": np.mean(widths), "median_w": np.median(widths),
        "mean_h": np.mean(heights), "median_h": np.median(heights),
        "median_area": np.median(areas), "mean_area": np.mean(areas),
        "median_short": np.median(short_edges),
        "median_aspect": np.median(aspects),
        "unique_sizes": unique_sizes,
    }

    # Area percentiles
    for p in [1, 10, 25, 50, 75, 90, 95, 99]:
        stats[f"area_p{p}"] = float(np.percentile(areas, p))

    # Short edge stats
    stats["pct_short_lt_256"] = sum(1 for s in short_edges if s < 256) / len(short_edges) * 100
    stats["pct_short_lt_96"] = sum(1 for s in short_edges if s < 96) / len(short_edges) * 100

    # Size distributions
    w_bins = [(0, 50, "0–49"), (50, 100, "50–99"), (100, 150, "100–149"), (150, 200, "150–199"),
              (200, 250, "200–249"), (250, 300, "250–299"), (300, 350, "300–349"),
              (350, 400, "350–399"), (400, 450, "400–449"), (450, 500, "450–499"),
              (500, 550, "500–549"), (550, 600, "550–599"), (600, 650, "600–649"),
              (650, 700, "650–699"), (700, 750, "700–749"), (750, 800, "750–799"),
              (800, 850, "800–849"), (850, 900, "850–899"), (900, 950, "900–949"),
              (950, 1000, "950–999")]
    stats["width_dist"] = make_hist(widths, w_bins)

    h_bins = [(0, 50, "0–49"), (50, 100, "50–99"), (100, 150, "100–149"),
              (150, 200, "150–199"), (200, 250, "200–249"), (250, 300, "250–299"),
              (300, 350, "300–349"), (350, 400, "350–399"), (400, 450, "400–449"),
              (450, 500, "450–499"), (500, 550, "500–549"), (550, 600, "550–599"),
              (600, 650, "600–649"), (650, 700, "650–699"), (700, 800, "700–799"),
              (800, 900, "800–899"), (900, 1000, "900–999"), (1000, 1200, "1000–1199"),
              (1200, 1400, "1200–1399"), (1400, 1600, "1400–1599"),
              (1600, 1800, "1600–1799"), (1800, 2000, "1800–1999")]
    stats["height_dist"] = make_hist(heights, h_bins)

    a_bins = [(0, 50, "0–49k"), (50, 100, "50–99k"), (100, 200, "100–199k"),
              (200, 300, "200–299k"), (300, 500, "300–499k"),
              (500, 1000, "500–999k")]
    stats["area_dist"] = make_hist(areas, a_bins)

    ar_bins = [(0, 0.5, "< 0.5 (极窄/竖长)"), (0.5, 0.67, "0.5–0.67"),
               (0.67, 0.8, "0.67–0.8"), (0.8, 0.9, "0.8–0.9"),
               (0.9, 1.1, "0.9–1.1 (正方形)"), (1.1, 1.3, "1.1–1.3"),
               (1.3, 1.5, "1.3–1.5"), (1.5, 2.0, "1.5–2.0")]
    stats["aspect_dist"] = make_hist(aspects, ar_bins)

    # Percentile buckets
    sorted_by_area = sorted(results, key=lambda r: r["area_kpx"])
    n = len(sorted_by_area)
    bucket_edges = [(0, 0.10, "P0–P10"), (0.10, 0.25, "P10–P25"), (0.25, 0.50, "P25–P50"),
                    (0.50, 0.75, "P50–P75"), (0.75, 0.90, "P75–P90"), (0.90, 1.00, "P90–P100")]
    buckets = []
    for lo, hi, label in bucket_edges:
        bucket = sorted_by_area[int(n * lo):int(n * hi)]
        N = len(bucket)
        mean_area_b = np.mean([r["area_kpx"] for r in bucket])
        mean_w_b = np.mean([r["orig_W"] for r in bucket])
        mean_h_b = np.mean([r["orig_H"] for r in bucket])
        mean_miou = np.mean([r["seg_mIoU"] for r in bucket])
        iou_upper = np.mean([r["per_class_IoU"][1] for r in bucket])
        iou_lower = np.mean([r["per_class_IoU"][2] for r in bucket])
        upper_prec = compute_precision(bucket, "upper")
        lower_prec = compute_precision(bucket, "lower")
        up_correct = sum(1 for r in bucket if r["upper_correct"])
        up_total = sum(1 for r in bucket if r["upper_gt"] != 0)
        lo_correct = sum(1 for r in bucket if r["lower_correct"])
        lo_total = sum(1 for r in bucket if r["lower_gt"] != 0)
        overall_prec = (up_correct + lo_correct) / max(1, up_total + lo_total)
        upper_top3 = top_colors_by_precision(bucket, "upper", top_k=3)
        lower_top3 = top_colors_by_precision(bucket, "lower", top_k=3)
        buckets.append({"label": label, "N": N, "mean_area": round(mean_area_b, 1),
                        "mean_w": int(round(mean_w_b)), "mean_h": int(round(mean_h_b)),
                        "mIoU": round(mean_miou, 4), "iou_upper": round(iou_upper, 4),
                        "iou_lower": round(iou_lower, 4),
                        "upper_precision": round(upper_prec, 3),
                        "lower_precision": round(lower_prec, 3),
                        "overall_precision": round(overall_prec, 3),
                        "upper_top3": upper_top3, "lower_top3": lower_top3})
    stats["buckets"] = buckets
    stats["best_miou"] = buckets[-1]["mIoU"]
    stats["miou_gain"] = buckets[-1]["mIoU"] - buckets[0]["mIoU"]

    # Threshold scans
    short_thr = []
    for t in [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]:
        below = [r for r in results if r["short_edge"] < t]
        above = [r for r in results if r["short_edge"] >= t]
        short_thr.append({"t": t, "below_pct": round(len(below) / n * 100, 1),
                          "below_N": len(below), "below_op": round(compute_overall_precision(below), 3),
                          "above_pct": round(len(above) / n * 100, 1),
                          "above_N": len(above), "above_op": round(compute_overall_precision(above), 3)})
    stats["short_thr"] = short_thr

    area_thr = []
    for t in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]:
        below = [r for r in results if r["area_kpx"] < t]
        above = [r for r in results if r["area_kpx"] >= t]
        area_thr.append({"t": t, "below_pct": round(len(below) / n * 100, 1),
                         "below_N": len(below), "below_op": round(compute_overall_precision(below), 3),
                         "above_pct": round(len(above) / n * 100, 1),
                         "above_N": len(above), "above_op": round(compute_overall_precision(above), 3)})
    stats["area_thr"] = area_thr

    # Global color precision
    stats["global_upper_prec"] = global_color_precision(results, "upper")
    stats["global_lower_prec"] = global_color_precision(results, "lower")

    return stats


def render_html(stats):
    """Render HTML using string replace instead of f-string to avoid JS brace conflicts."""
    # Convert numpy values to Python native types
    def conv(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    # Build data JSON for the JavaScript sections
    js_data = {
        "buckets": stats["buckets"],
        "short_thr": stats["short_thr"],
        "area_thr": stats["area_thr"],
        "global_prec": {
            "upper": stats["global_upper_prec"],
            "lower": stats["global_lower_prec"],
        }
    }
    data_json = json.dumps(js_data, ensure_ascii=False)

    # Distribution table data
    wd_json = json.dumps([(r[0] + ' px', r[1], round(r[2], 1)) for r in stats["width_dist"]], ensure_ascii=False)
    hd_json = json.dumps([(r[0] + ' px', r[1], round(r[2], 1)) for r in stats["height_dist"]], ensure_ascii=False)
    ad_json = json.dumps([(r[0], r[1], round(r[2], 1)) for r in stats["area_dist"]], ensure_ascii=False)
    ard_json = json.dumps([(r[0], r[1], round(r[2], 1)) for r in stats["aspect_dist"]], ensure_ascii=False)

    # Area percentiles
    area_p_items = []
    for p in [1, 10, 25, 50, 75, 90, 95, 99]:
        v = stats[f"area_p{p}"]
        if v >= 1000:
            area_p_items.append(f'<div class="kv-card"><div class="kv-val">{v/1000:.1f}k</div><div class="kv-lbl">P{p}</div></div>')
        else:
            area_p_items.append(f'<div class="kv-card"><div class="kv-val">{v:.1f} k</div><div class="kv-lbl">P{p}</div></div>')
    area_p_html = ''.join(area_p_items)

    # Summary stats
    bw = stats["buckets"]
    miou_range_str = f"{bw[0]['mIoU']:.4f} (P0-10) → {bw[-1]['mIoU']:.4f} (P90-100)"
    prec_range_str = f"{bw[0]['overall_precision']:.3f} → {bw[-1]['overall_precision']:.3f}"

    # Read CSS from a separate file (or inline)
    css = get_css()

    # Use template with $$ placeholders to avoid f-string/JS conflicts
    html = get_html_template()
    html = html.replace("$$CSS$$", css)
    html = html.replace("$$TOTAL$$", f"{stats['total']:,}")
    html = html.replace("$$PCT_UPSAMPLE$$", f"{stats['pct_short_lt_256']:.1f}")
    html = html.replace("$$BEST_MIOU$$", f"{stats['best_miou']:.2f}")
    html = html.replace("$$MIOU_GAIN$$", f"+{stats['miou_gain']*100:.0f}%")
    html = html.replace("$$MEAN_W$$", f"{int(stats['mean_w'])}")
    html = html.replace("$$MEAN_H$$", f"{int(stats['mean_h'])}")
    html = html.replace("$$MEDIAN_AREA$$", f"{stats['median_area']:.1f}")
    html = html.replace("$$MEDIAN_W$$", f"{int(stats['median_w'])}")
    html = html.replace("$$MEDIAN_H$$", f"{int(stats['median_h'])}")
    html = html.replace("$$MEDIAN_SHORT$$", f"{int(stats['median_short'])}")
    html = html.replace("$$MEDIAN_ASPECT$$", f"{stats['median_aspect']:.3f}")
    html = html.replace("$$ASPECT_RATIO_STR$$", f"1:{1/stats['median_aspect']:.1f}")
    html = html.replace("$$UNIQUE_SIZES$$", f"{stats['unique_sizes']:,}")
    html = html.replace("$$AREA_PERCENTILES$$", area_p_html)
    html = html.replace("$$MIOU_RANGE$$", miou_range_str)
    html = html.replace("$$PREC_RANGE$$", prec_range_str)
    html = html.replace("$$PCT_SHORT_LT_96$$", f"{stats['pct_short_lt_96']:.1f}")

    # Bucket labels for degradation charts
    html = html.replace("$$B0_AREA$$", f"{bw[0]['mean_area']:.0f}")
    html = html.replace("$$B5_AREA$$", f"{bw[-1]['mean_area']:.0f}")
    html = html.replace("$$MEDIAN_AREA_LABEL$$", f"{stats['median_area']:.0f}")

    # Data JSON
    html = html.replace("$$DATA_JSON$$", data_json)
    html = html.replace("$$WD_JSON$$", wd_json)
    html = html.replace("$$HD_JSON$$", hd_json)
    html = html.replace("$$AD_JSON$$", ad_json)
    html = html.replace("$$ARD_JSON$$", ard_json)

    return html


def get_css():
    return """/* ── Token System ── */
:root {
  --bg: #080b11;
  --bg-card: #0f131e;
  --bg-card-alt: #141926;
  --bg-table-header: #1a1f30;
  --text: #e2e6ef;
  --text-secondary: #8893a8;
  --text-muted: #545e73;
  --accent: #f0b429;
  --accent-dim: rgba(240,180,41,0.15);
  --cyan: #22d3ee;
  --red: #f87171;
  --green: #4ade80;
  --border: #1e2535;
  --border-light: #2a3145;
  --radius: 8px;
  --radius-sm: 4px;
  --font-display: 'Georgia', 'Noto Serif SC', 'Songti SC', serif;
  --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  --font-mono: 'SF Mono', 'JetBrains Mono', 'Consolas', 'Cascadia Code', 'Source Code Pro', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 15px; background: var(--bg); color: var(--text); font-family: var(--font-body); -webkit-font-smoothing: antialiased; }
body { min-height: 100vh; line-height: 1.6; }
.container { max-width: 1120px; margin: 0 auto; padding: 0 24px; }
.hero { padding: 72px 0 56px; border-bottom: 1px solid var(--border); position: relative; }
.hero::after { content: ''; position: absolute; bottom: -1px; left: 0; width: 120px; height: 3px; background: var(--accent); border-radius: 3px 3px 0 0; }
.hero-label { font-family: var(--font-mono); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; color: var(--text-muted); margin-bottom: 16px; }
.hero h1 { font-family: var(--font-display); font-size: 2.6rem; font-weight: 400; line-height: 1.25; color: var(--text); margin-bottom: 12px; letter-spacing: -0.02em; }
.hero h1 em { font-style: normal; color: var(--accent); }
.hero .subtitle { font-size: 1.05rem; color: var(--text-secondary); max-width: 680px; }
.stat-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border); margin: 48px 0; border-radius: var(--radius); overflow: hidden; }
.stat-item { background: var(--bg-card); padding: 24px 28px; text-align: center; }
.stat-value { font-family: var(--font-display); font-size: 2rem; color: var(--accent); line-height: 1.1; margin-bottom: 4px; }
.stat-label { font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.section { margin: 56px 0; }
.section-header { margin-bottom: 32px; }
.section-number { font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent); letter-spacing: 0.15em; margin-bottom: 8px; }
.section-header h2 { font-family: var(--font-display); font-size: 1.7rem; font-weight: 400; color: var(--text); margin-bottom: 8px; letter-spacing: -0.01em; }
.section-header p { color: var(--text-secondary); font-size: 0.95rem; max-width: 640px; }
.card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; margin-bottom: 24px; }
.card-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
.card-header h3 { font-family: var(--font-display); font-size: 1.15rem; font-weight: 400; color: var(--text); }
.card-badge { font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent); background: var(--accent-dim); padding: 3px 10px; border-radius: 100px; }
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
thead th { font-family: var(--font-mono); font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); text-align: right; padding: 10px 10px; background: var(--bg-table-header); border-bottom: 2px solid var(--border); white-space: nowrap; }
thead th:first-child { text-align: left; }
thead th.col-ctr { text-align: center; }
tbody td { padding: 9px 10px; text-align: right; border-bottom: 1px solid var(--border); font-family: var(--font-body); white-space: nowrap; }
tbody td:first-child { text-align: left; font-family: var(--font-mono); color: var(--text-secondary); font-size: 0.78rem; }
tbody tr:hover { background: var(--bg-card-alt); }
tbody tr:last-child td { border-bottom: none; }
.dist-bar-wrap { display: flex; align-items: center; gap: 6px; justify-content: flex-end; }
.dist-bar { height: 8px; border-radius: 4px; transition: width 0.4s; }
.dist-pct { font-size: 0.75rem; min-width: 48px; text-align: right; font-weight: 500; }
.perf-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.perf-excellent { background: var(--green); box-shadow: 0 0 6px rgba(74,222,128,0.4); }
.perf-good { background: #22c55e; }
.perf-fair { background: var(--accent); box-shadow: 0 0 6px rgba(240,180,41,0.4); }
.perf-poor { background: #fb923c; }
.perf-bad { background: var(--red); box-shadow: 0 0 6px rgba(248,113,113,0.4); }
.mini-bar { display: inline-block; height: 6px; border-radius: 3px; vertical-align: middle; margin-right: 6px; }
.bar-high { background: var(--green); }
.bar-mid { background: var(--accent); }
.bar-low { background: var(--red); }
.color-tag { display: inline-block; padding: 1px 8px; border-radius: 100px; font-size: 0.72rem; font-weight: 500; border: 1px solid; }
.color-black { color: #e2e8f0; background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.15); }
.color-white { color: #e2e8f0; background: rgba(255,255,255,0.12); border-color: rgba(255,255,255,0.25); }
.color-gray { color: #9ca3af; background: rgba(156,163,175,0.12); border-color: rgba(156,163,175,0.2); }
.color-red { color: #f87171; background: rgba(248,113,113,0.1); border-color: rgba(248,113,113,0.2); }
.color-yellow { color: #facc15; background: rgba(250,204,21,0.1); border-color: rgba(250,204,21,0.2); }
.color-green { color: #4ade80; background: rgba(74,222,128,0.1); border-color: rgba(74,222,128,0.2); }
.color-blue { color: #60a5fa; background: rgba(96,165,250,0.1); border-color: rgba(96,165,250,0.2); }
.color-purple { color: #c084fc; background: rgba(192,132,252,0.1); border-color: rgba(192,132,252,0.2); }
.color-pink { color: #f472b6; background: rgba(244,114,182,0.1); border-color: rgba(244,114,182,0.2); }
.color-orange { color: #fb923c; background: rgba(251,146,60,0.1); border-color: rgba(251,146,60,0.2); }
.color-brown { color: #a78b6f; background: rgba(167,139,111,0.1); border-color: rgba(167,139,111,0.2); }
.deg-chart { display: flex; align-items: flex-end; gap: 4px; height: 150px; padding: 0 8px; margin: 20px 0; }
.deg-bar { flex: 1; border-radius: 3px 3px 0 0; position: relative; cursor: default; min-width: 20px; transition: opacity 0.2s; }
.deg-bar:hover { opacity: 0.85; }
.deg-bar .bar-val { position: absolute; top: -16px; left: 50%; transform: translateX(-50%); font-family: var(--font-body); font-size: 0.62rem; font-weight: 700; white-space: nowrap; }
.deg-bar .bar-lbl { position: absolute; bottom: -20px; left: 50%; transform: translateX(-50%); font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-muted); }
.threshold-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 20px; }
.th-card { background: var(--bg-card-alt); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; position: relative; overflow: hidden; transition: border-color 0.2s; }
.th-card:hover { border-color: var(--border-light); }
.th-card.recommended { border-color: var(--accent); border-width: 1.5px; }
.th-card.recommended::before { content: '推荐'; position: absolute; top: 14px; right: 14px; font-family: var(--font-mono); font-size: 0.65rem; background: var(--accent); color: var(--bg); padding: 2px 10px; border-radius: 100px; font-weight: 600; letter-spacing: 0.06em; }
.th-card h4 { font-family: var(--font-display); font-size: 1rem; font-weight: 400; color: var(--text); margin-bottom: 16px; }
.th-rule { font-family: var(--font-mono); font-size: 0.78rem; color: var(--cyan); margin-bottom: 6px; padding: 6px 10px; background: rgba(34,211,238,0.06); border-radius: var(--radius-sm); }
.th-stat { display: flex; justify-content: space-between; font-size: 0.8rem; padding: 6px 0; border-bottom: 1px solid var(--border); color: var(--text-secondary); }
.th-stat:last-child { border-bottom: none; }
.th-stat .val { color: var(--text); font-family: var(--font-body); font-weight: 600; }
.reco-box { background: var(--accent-dim); border: 1px solid rgba(240,180,41,0.25); border-radius: var(--radius); padding: 24px 28px; margin-top: 32px; }
.reco-box h4 { font-family: var(--font-display); font-size: 1.1rem; font-weight: 400; color: var(--accent); margin-bottom: 12px; }
.reco-box p { color: var(--text-secondary); font-size: 0.9rem; line-height: 1.7; }
.reco-box code { font-family: var(--font-mono); font-size: 0.82rem; color: var(--cyan); background: rgba(34,211,238,0.08); padding: 1px 6px; border-radius: 3px; }
.callout { background: var(--bg-card-alt); border-left: 3px solid var(--cyan); padding: 16px 20px; border-radius: 0 var(--radius-sm) var(--radius-sm) 0; margin-bottom: 20px; }
.callout p { font-size: 0.88rem; color: var(--text-secondary); line-height: 1.65; }
.callout strong { color: var(--text); }
.kv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
.kv-card { background: var(--bg-card-alt); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; text-align: center; }
.kv-card .kv-val { font-family: var(--font-display); font-size: 1.6rem; color: var(--accent); line-height: 1.2; margin-bottom: 4px; }
.kv-card .kv-lbl { font-size: 0.75rem; color: var(--text-muted); }
.footer { margin-top: 72px; padding: 32px 0; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.8rem; display: flex; justify-content: space-between; }
.footer span { font-family: var(--font-mono); }
thead th.sep-col { background: var(--border); padding: 0 !important; width: 2px; min-width: 2px; }
@media (max-width: 768px) { .hero h1 { font-size: 1.8rem; } .stat-strip { grid-template-columns: repeat(2, 1fr); } .threshold-grid { grid-template-columns: 1fr; } .section-header h2 { font-size: 1.35rem; } .card { padding: 18px; } .kv-grid { grid-template-columns: repeat(2, 1fr); } }
@media print { :root { --bg: #fff; --bg-card: #fff; --bg-card-alt: #f8f9fa; --text: #111; --text-secondary: #444; } .hero::after { background: #000; } }"""


def get_html_template():
    return r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>图片尺寸 vs 模型性能分析 — PP-LiteSeg STDC2 RGB-Only 256px · iter_22000 · train.txt</title>
<style>
$$CSS$$
</style>
</head>
<body>
<div class="container">

<!-- HERO -->
<header class="hero">
  <div class="hero-label">Analysis Report · 2026-06-26</div>
  <h1>图片尺寸如何影响<em>分割与颜色分类</em>性能</h1>
  <p class="subtitle">
    基于 PP-LiteSeg STDC2 RGB-Only 双头模型（3ch·256px 输入·iter_22000）在 $$TOTAL$$ 张训练集上的推理结果，
    系统分析图片像素尺寸与模型效果之间的定量关系，并给出摄像头部署时的属性预测<strong>尺寸准入阈值</strong>建议。
  </p>
</header>

<!-- KEY STATS -->
<div class="stat-strip">
  <div class="stat-item"><div class="stat-value">$$TOTAL$$</div><div class="stat-label">训练集总量</div></div>
  <div class="stat-item"><div class="stat-value">$$PCT_UPSAMPLE$$%</div><div class="stat-label">图片需上采样</div></div>
  <div class="stat-item"><div class="stat-value">$$BEST_MIOU$$</div><div class="stat-label">最佳 mIoU（大图）</div></div>
  <div class="stat-item"><div class="stat-value">$$MIOU_GAIN$$</div><div class="stat-label">mIoU 尺寸增益</div></div>
</div>

<!-- PART 0: DATASET SIZE STATISTICS -->
<section class="section" id="dataset-stats">
  <div class="section-header">
    <div class="section-number">PART 0</div>
    <h2>数据集图片尺寸统计</h2>
    <p>对 train.txt 中全部 $$TOTAL$$ 张图片的像素尺寸进行全面统计。</p>
  </div>

  <div class="kv-grid">
    <div class="kv-card"><div class="kv-val">$$MEAN_W$$ px</div><div class="kv-lbl">均值宽度</div></div>
    <div class="kv-card"><div class="kv-val">$$MEAN_H$$ px</div><div class="kv-lbl">均值高度</div></div>
    <div class="kv-card"><div class="kv-val">$$MEDIAN_AREA$$ kpx</div><div class="kv-lbl">中位数面积 (≈ $$MEDIAN_W$$×$$MEDIAN_H$$)</div></div>
    <div class="kv-card"><div class="kv-val">$$MEDIAN_SHORT$$ px</div><div class="kv-lbl">中位数短边</div></div>
    <div class="kv-card"><div class="kv-val">$$MEDIAN_ASPECT$$</div><div class="kv-lbl">中位数宽高比 (≈ $$ASPECT_RATIO_STR$$ 竖长)</div></div>
    <div class="kv-card"><div class="kv-val">$$UNIQUE_SIZES$$ 种</div><div class="kv-lbl">独特 (W,H) 组合</div></div>
  </div>

  <!-- Distribution tables -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:24px;">
    <div class="card" style="margin-bottom:0;">
      <div class="card-header"><h3>宽度分布</h3><span class="card-badge">每 50px 一档</span></div>
      <div class="table-wrap"><table><thead><tr><th>档位</th><th>数量</th><th class="col-ctr">占比</th></tr></thead><tbody id="width-dist"></tbody></table></div>
    </div>
    <div class="card" style="margin-bottom:0;">
      <div class="card-header"><h3>高度分布</h3><span class="card-badge">每 50px 一档</span></div>
      <div class="table-wrap"><table><thead><tr><th>档位</th><th>数量</th><th class="col-ctr">占比</th></tr></thead><tbody id="height-dist"></tbody></table></div>
    </div>
    <div class="card" style="margin-bottom:0;">
      <div class="card-header"><h3>面积分布</h3><span class="card-badge">kpx</span></div>
      <div class="table-wrap"><table><thead><tr><th>档位</th><th>数量</th><th class="col-ctr">占比</th></tr></thead><tbody id="area-dist"></tbody></table></div>
    </div>
    <div class="card" style="margin-bottom:0;">
      <div class="card-header"><h3>宽高比分布</h3><span class="card-badge">W/H</span></div>
      <div class="table-wrap"><table><thead><tr><th>档位</th><th>数量</th><th class="col-ctr">占比</th></tr></thead><tbody id="aspect-dist"></tbody></table></div>
    </div>
  </div>

  <!-- Area percentiles -->
  <div class="card" style="margin-top:24px;">
    <div class="card-header"><h3>面积百分位数</h3><span class="card-badge">kpx</span></div>
    <div class="kv-grid">$$AREA_PERCENTILES$$</div>
  </div>
</section>

<!-- PART 1: PERCENTILE TABLE -->
<section class="section" id="percentile-table">
  <div class="section-header">
    <div class="section-number">PART 1</div>
    <h2>面积百分位 vs. 模型性能（Precision 视角）</h2>
    <p>将训练集按图片面积 (kpx) 从低到高排序后分档。<strong>Precision（精确率）= 正确预测数 / 总预测数</strong>。每种颜色至少需 5 次预测才计入 Top-3 排名。</p>
  </div>
  <div class="callout"><p><strong>为什么用 Precision？</strong> 在摄像头部署场景中，Precision 直接回答「这次预测可信吗？」。低 Precision 意味着大量错误预测，比低 Recall 的影响更直接。</p></div>

  <div class="card">
    <div class="card-header"><h3>面积分档详细指标（Precision）</h3><span class="card-badge">N = $$TOTAL$$</span></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>面积分位</th><th class="col-ctr">N</th><th>均值面积</th><th>均值尺寸</th><th>mIoU</th><th>上衣 Prec</th><th>下衣 Prec</th><th>总色 Prec</th><th style="min-width:280px">上衣 Top-3 颜色 (Prec)</th></tr></thead>
        <tbody id="buckets-tbody"></tbody>
      </table>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:24px;">
    <div class="card" style="margin-bottom:0;">
      <div class="card-header"><h3>全局上衣颜色 Precision 排名</h3></div>
      <div class="table-wrap"><table><thead><tr><th>颜色</th><th>Precision</th><th>预测次数</th></tr></thead><tbody id="global-upper-prec"></tbody></table></div>
    </div>
    <div class="card" style="margin-bottom:0;">
      <div class="card-header"><h3>全局下衣颜色 Precision 排名</h3></div>
      <div class="table-wrap"><table><thead><tr><th>颜色</th><th>Precision</th><th>预测次数</th></tr></thead><tbody id="global-lower-prec"></tbody></table></div>
    </div>
  </div>
</section>

<!-- PART 2: DEGRADATION CURVES -->
<section class="section" id="degradation">
  <div class="section-header"><div class="section-number">PART 2</div><h2>性能退化曲线</h2><p>面积从极小到大，mIoU 和颜色 Precision 的单调递增趋势。</p></div>
  <div class="card">
    <div class="card-header"><h3>mIoU 按面积分位</h3><span class="card-badge">面积越大 → 分割越好</span></div>
    <div class="deg-chart" id="chart-mIoU"></div>
    <div style="display:flex;justify-content:space-between;padding:28px 8px 0;font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono);">
      <span>← 极小图 $$B0_AREA$$ kpx</span><span>中位数 $$MEDIAN_AREA_LABEL$$ kpx</span><span>大图 $$B5_AREA$$ kpx →</span>
    </div>
  </div>
  <div class="card">
    <div class="card-header"><h3>颜色 Precision 按面积分位</h3><span class="card-badge">颜色对尺寸更敏感</span></div>
    <div class="deg-chart" id="chart-prec"></div>
    <div style="display:flex;justify-content:space-between;padding:28px 8px 0;font-size:0.7rem;color:var(--text-muted);font-family:var(--font-mono);">
      <span>← 极小图 $$B0_AREA$$ kpx</span><span>中位数 $$MEDIAN_AREA_LABEL$$ kpx</span><span>大图 $$B5_AREA$$ kpx →</span>
    </div>
  </div>
</section>

<!-- PART 3: DEPLOYMENT THRESHOLDS -->
<section class="section" id="thresholds">
  <div class="section-header"><div class="section-number">PART 3</div><h2>部署阈值建议（Precision 基准）</h2><p>基于 Precision 给出<strong>短边、总面积</strong>二维度的准入阈值。</p></div>

  <div class="card">
    <div class="card-header"><h3>短边 (min(W,H)) 阈值扫描</h3><span class="card-badge">核心维度：决定上采样倍数</span></div>
    <div class="table-wrap"><table><thead><tr><th>短边阈值</th><th colspan="3" style="text-align:center;color:var(--red);">过滤掉 (&lt; 阈值)</th><th class="sep-col"></th><th colspan="3" style="text-align:center;color:var(--green);">保留 (≥ 阈值)</th></tr><tr><th></th><th>占比</th><th>N</th><th>总色 Prec</th><th class="sep-col"></th><th>占比</th><th>N</th><th>总色 Prec</th></tr></thead><tbody id="short-tbody"></tbody></table></div>
  </div>
  <div class="card">
    <div class="card-header"><h3>总面积 (kpx) 阈值扫描</h3><span class="card-badge">综合维度</span></div>
    <div class="table-wrap"><table><thead><tr><th>面积阈值</th><th colspan="3" style="text-align:center;color:var(--red);">过滤掉 (&lt; 阈值)</th><th class="sep-col"></th><th colspan="3" style="text-align:center;color:var(--green);">保留 (≥ 阈值)</th></tr><tr><th></th><th>占比</th><th>N</th><th>总色 Prec</th><th class="sep-col"></th><th>占比</th><th>N</th><th>总色 Prec</th></tr></thead><tbody id="area-tbody"></tbody></table></div>
  </div>

  <div class="section-header" style="margin-top:40px;"><h2>推荐阈值方案</h2><p>综合短边、面积和 Precision 拐点分析，提供三级阈值。推荐使用<strong>平衡方案</strong>作为默认。</p></div>
  <div class="threshold-grid" id="thresh-cards"></div>

  <div class="reco-box">
    <h4>💡 实施建议</h4>
    <p>1. <strong>默认使用平衡方案</strong>：当 crop 的 <code>short &lt; 96 px</code> <strong>且</strong> <code>area &lt; 25 kpx</code> 时，跳过颜色属性预测。<br>
    2. <strong>短边是最强的单一判据</strong>——它直接决定上采样倍数。短边 &lt; 64px 时上采样 &ge; 4×，Precision 显著下降。<br>
    3. 过滤后，剩余图片维持正常属性预测通路。<br>
    4. <strong>⚠️ 本报告基于训练集 (train.txt) 推理</strong>，模型见过这些数据，指标偏高；实际部署应以验证集评估为准。</p>
  </div>
</section>

<!-- APPENDIX -->
<section class="section" id="notes">
  <div class="section-header"><div class="section-number">APPENDIX</div><h2>指标说明与方法备注</h2></div>
  <div class="callout"><p><strong>Precision 计算方式</strong>：<code>Precision = 正确预测数 / 总预测数</code>。对每张图片，模型对上衣和下衣各输出一个颜色预测。</p></div>
  <div class="callout"><p><strong>模型输入配置</strong>：PP-LiteSeg STDC2 · 双线性插值 resize 至 256×256 · 3 通道 RGB only · 双头输出（分割 8 类 + 颜色 12 类 × 2）· iter_22000 · optC 从头训练。</p></div>
  <div class="callout"><p><strong>⚠️ 本报告基于训练集</strong>：推理在 train.txt ($$TOTAL$$ 张) 上进行。各项指标可能显著高于验证集真实表现。本分析旨在揭示尺寸与性能的趋势关系，具体阈值数值应使用验证集重新校准。约 $$PCT_SHORT_LT_96$$% 图片短边 &lt; 96px，上采样 &ge; 2.7×。</p></div>
</section>

<div class="footer">
  <span>PP-LiteSeg STDC2 RGB-Only · iter_22000 · train.txt</span>
  <span>Generated 2026-06-26</span>
</div>

</div>

<script>
var D = $$DATA_JSON$$;

function cPill(c, v, n) {
  return '<span class="color-tag color-'+c+'">'+c+'</span>&nbsp;<span style="font-size:0.78rem;font-weight:600;">'+(v*100).toFixed(1)+'%</span>&nbsp;<span style="font-size:0.65rem;color:var(--text-muted);">n='+n+'</span>';
}

// Buckets table
(function(){
  var tbody = document.getElementById('buckets-tbody');
  var rows = D.buckets.map(function(b){
    var dotC = b.mIoU >= 0.55 ? 'perf-excellent' : b.mIoU >= 0.48 ? 'perf-good' : b.mIoU >= 0.40 ? 'perf-fair' : b.mIoU >= 0.33 ? 'perf-poor' : 'perf-bad';
    return '<tr><td><span class="perf-dot '+dotC+'"></span>'+b.label+'</td><td style="text-align:center">'+b.N+'</td><td>'+b.mean_area+' kpx</td><td>'+b.mean_w+'×'+b.mean_h+'</td><td><strong>'+b.mIoU.toFixed(4)+'</strong></td><td>'+b.upper_precision.toFixed(3)+'</td><td>'+b.lower_precision.toFixed(3)+'</td><td><strong>'+b.overall_precision.toFixed(3)+'</strong></td><td style="font-size:0.75rem;">'+b.upper_top3.map(function(c){return cPill(c.color,c.prec,c.n);}).join('<br>')+'</td></tr>';
  }).join('');
  tbody.innerHTML = rows;
})();

// Threshold tables
function thrTab(id, data, unit) {
  var h = data.map(function(r){
    var clsB = r.below_op >= 0.55 ? 'color:var(--green)' : r.below_op >= 0.52 ? 'color:var(--accent)' : 'color:var(--red)';
    var clsA = r.above_op >= 0.55 ? 'color:var(--green)' : r.above_op >= 0.52 ? 'color:var(--accent)' : 'color:var(--red)';
    return '<tr><td>&lt; '+r.t+' '+unit+'</td><td>'+r.below_pct+'%</td><td>'+r.below_N+'</td><td><strong style="'+clsB+'">'+(r.below_op*100).toFixed(1)+'%</strong></td><td class="sep-col"></td><td>'+r.above_pct+'%</td><td>'+r.above_N+'</td><td><strong style="'+clsA+'">'+(r.above_op*100).toFixed(1)+'%</strong></td></tr>';
  }).join('');
  document.getElementById(id).innerHTML = h;
}
thrTab('short-tbody', D.short_thr, 'px');
thrTab('area-tbody', D.area_thr, 'kpx');

// Global precision tables
(function(){
  var hu = D.global_prec.upper.map(function(c,i){
    var dot = i<3 ? 'perf-excellent' : i<6 ? 'perf-good' : i<9 ? 'perf-fair' : 'perf-bad';
    return '<tr><td><span class="perf-dot '+dot+'"></span><span class="color-tag color-'+c.color+'">'+c.color_zh+'</span></td><td><strong>'+(c.prec*100).toFixed(1)+'%</strong></td><td>'+c.n_pred+'</td></tr>';
  }).join('');
  document.getElementById('global-upper-prec').innerHTML = hu;
  var hl = D.global_prec.lower.map(function(c,i){
    var dot = i<3 ? 'perf-excellent' : i<6 ? 'perf-good' : i<9 ? 'perf-fair' : 'perf-bad';
    return '<tr><td><span class="perf-dot '+dot+'"></span><span class="color-tag color-'+c.color+'">'+c.color_zh+'</span></td><td><strong>'+(c.prec*100).toFixed(1)+'%</strong></td><td>'+c.n_pred+'</td></tr>';
  }).join('');
  document.getElementById('global-lower-prec').innerHTML = hl;
})();

// Degradation charts
(function(){
  var colors = ['#f87171','#fb923c','#fbbf24','#fbbf24','#a3e635','#4ade80'];
  var mH = 135;
  var cm = document.getElementById('chart-mIoU');
  D.buckets.forEach(function(b,i){
    var fullRange = D.buckets[5].mIoU - D.buckets[0].mIoU;
    var h = Math.round((b.mIoU - D.buckets[0].mIoU) / Math.max(fullRange, 0.001) * mH);
    var bar = document.createElement('div');
    bar.className = 'deg-bar';
    bar.style.height = Math.max(h,8)+'px';
    bar.style.background = 'linear-gradient(180deg, '+colors[i]+', '+colors[i]+'88)';
    bar.title = b.label+': mIoU = '+b.mIoU.toFixed(4);
    bar.innerHTML = '<span class="bar-val" style="color:'+colors[i]+'">'+b.mIoU.toFixed(3)+'</span><span class="bar-lbl">'+b.label+'</span>';
    cm.appendChild(bar);
  });
  var cp = document.getElementById('chart-prec');
  D.buckets.forEach(function(b,i){
    var fullRange = D.buckets[5].overall_precision - D.buckets[0].overall_precision;
    var h = Math.round((b.overall_precision - D.buckets[0].overall_precision) / Math.max(fullRange, 0.001) * mH);
    var bar = document.createElement('div');
    bar.className = 'deg-bar';
    bar.style.height = Math.max(h,8)+'px';
    bar.style.background = 'linear-gradient(180deg, '+colors[i]+', '+colors[i]+'88)';
    bar.title = b.label+': Precision = '+(b.overall_precision*100).toFixed(1)+'%';
    bar.innerHTML = '<span class="bar-val" style="color:'+colors[i]+'">'+(b.overall_precision*100).toFixed(0)+'%</span><span class="bar-lbl">'+b.label+'</span>';
    cp.appendChild(bar);
  });
})();

// Size distribution tables
(function(){
  var wd = $$WD_JSON$$;
  var maxNw = Math.max.apply(null, wd.map(function(r){return r[1];}));
  document.getElementById('width-dist').innerHTML = wd.map(function(r){
    return '<tr><td>'+r[0]+'</td><td>'+r[1].toLocaleString()+'</td><td class="col-ctr"><div class="dist-bar-wrap"><div class="dist-bar" style="width:'+(r[1]/maxNw*120)+'px;background:var(--accent);"></div><span class="dist-pct">'+r[2].toFixed(1)+'%</span></div></td></tr>';
  }).join('');

  var hd = $$HD_JSON$$;
  var maxNh = Math.max.apply(null, hd.map(function(r){return r[1];}));
  document.getElementById('height-dist').innerHTML = hd.map(function(r){
    return '<tr><td>'+r[0]+'</td><td>'+r[1].toLocaleString()+'</td><td class="col-ctr"><div class="dist-bar-wrap"><div class="dist-bar" style="width:'+(r[1]/maxNh*120)+'px;background:var(--cyan);"></div><span class="dist-pct">'+r[2].toFixed(1)+'%</span></div></td></tr>';
  }).join('');

  var ad = $$AD_JSON$$;
  var maxNa = Math.max.apply(null, ad.map(function(r){return r[1];}));
  document.getElementById('area-dist').innerHTML = ad.map(function(r){
    return '<tr><td>'+r[0]+'</td><td>'+r[1].toLocaleString()+'</td><td class="col-ctr"><div class="dist-bar-wrap"><div class="dist-bar" style="width:'+(r[1]/maxNa*120)+'px;background:var(--green);"></div><span class="dist-pct">'+r[2].toFixed(1)+'%</span></div></td></tr>';
  }).join('');

  var ard = $$ARD_JSON$$;
  var maxAr = Math.max.apply(null, ard.map(function(r){return r[1];}));
  document.getElementById('aspect-dist').innerHTML = ard.map(function(r){
    return '<tr><td>'+r[0]+'</td><td>'+r[1].toLocaleString()+'</td><td class="col-ctr"><div class="dist-bar-wrap"><div class="dist-bar" style="width:'+(r[1]/maxAr*120)+'px;background:#a78bfa;"></div><span class="dist-pct">'+r[2].toFixed(1)+'%</span></div></td></tr>';
  }).join('');
})();

// Threshold cards
(function(){
  var sh96 = D.short_thr.find(function(r){return r.t==96;}) || D.short_thr[4];
  var ar25 = D.area_thr.find(function(r){return r.t==25;}) || D.area_thr[4];
  var sh80 = D.short_thr.find(function(r){return r.t==80;}) || D.short_thr[3];
  var ar15 = D.area_thr.find(function(r){return r.t==15;}) || D.area_thr[2];
  var sh128 = D.short_thr.find(function(r){return r.t==128;}) || D.short_thr[6];
  var ar40 = D.area_thr.find(function(r){return r.t==40;}) || D.area_thr[6];

  var cards = [
    {title: '🛡️ 保守方案', rules: ['short ≥ 80 px', 'area ≥ 15 kpx'], filterPct: (sh80.below_pct).toFixed(1)+'%', prec: (sh80.above_op*100).toFixed(1)+'%', scene: '高可靠性需求'},
    {title: '⚖️ 平衡方案', rules: ['short ≥ 96 px', 'area ≥ 25 kpx'], filterPct: (sh96.below_pct).toFixed(1)+'%', prec: (sh96.above_op*100).toFixed(1)+'%', scene: '一般生产环境 ✓', recommended: true},
    {title: '🚀 激进方案', rules: ['short ≥ 128 px', 'area ≥ 40 kpx'], filterPct: (sh128.below_pct).toFixed(1)+'%', prec: (sh128.above_op*100).toFixed(1)+'%', scene: '最大化覆盖'}
  ];

  document.getElementById('thresh-cards').innerHTML = cards.map(function(c){
    var cls = c.recommended ? 'th-card recommended' : 'th-card';
    return '<div class="'+cls+'"><h4>'+c.title+'</h4>'+
      c.rules.map(function(r){return '<div class="th-rule">'+r+'</div>';}).join('')+
      '<div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);margin:2px 0 6px 10px;">且关系</div>'+
      '<div class="th-stat"><span>过滤比例</span><span class="val">~'+c.filterPct+'</span></div>'+
      '<div class="th-stat"><span>过滤后颜色 Prec</span><span class="val" style="color:var(--green)">~'+c.prec+'</span></div>'+
      '<div class="th-stat"><span>适用场景</span><span class="val" style="color:var(--text-secondary);font-size:0.73rem;">'+c.scene+'</span></div>'+
    '</div>';
  }).join('');
})();
</script>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    print(f"Loaded {len(results)} images")

    print("Computing statistics...")
    stats = compute_all_stats(results)

    print("Rendering HTML...")
    html = render_html(stats)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to {args.output}")
    bw = stats["buckets"]
    print(f"  Images: {stats['total']:,}")
    print(f"  mIoU range: {bw[0]['mIoU']:.4f} (P0-10) → {bw[-1]['mIoU']:.4f} (P90-100)")
    print(f"  Color prec range: {bw[0]['overall_precision']:.3f} → {bw[-1]['overall_precision']:.3f}")


if __name__ == "__main__":
    main()
