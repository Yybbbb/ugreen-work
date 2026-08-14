#!/usr/bin/env python3
"""
Generate a self-contained HTML report with matplotlib-rendered charts.

Usage:
  cd /data1/work/MichaelYu/segment-color/statistics
  python gen_detection_html.py
"""

import base64
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Constants ─────────────────────────────────────────────────────────────────
SEG_CLASS_NAMES = ["hat", "glasses", "mask", "hair"]

SEG_COLORS_RGB = {
    0: (0,   0,   0),     # background – black
    1: (255, 0,   0),     # upper      – red
    2: (0,   255, 0),     # lower      – lime
    3: (0,   0,   255),   # dress      – blue
    4: (255, 255, 0),     # hat        – yellow
    5: (255, 0,   255),   # glasses    – magenta
    6: (0,   128, 128),   # mask       – teal
    7: (255, 255, 255),   # hair       – white
}

DISPLAY_NAMES = {
    "background": "Background",
    "upper": "Upper",
    "lower": "Lower",
    "dress": "Dress",
    "hat": "Hat",
    "glasses": "Glasses",
    "mask": "Mask",
    "hair": "Hair",
}

NUM_BUCKETS = 10
FIGSIZE = (5.2, 3.6)
DPI = 120

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": DPI,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(csv_path):
    """Load per_image_detection.csv as list of dicts with numeric values."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"image_rel": row["image_rel"]}
            for n in SEG_CLASS_NAMES:
                for suffix in ["gt_present", "pred_present", "tp", "fp", "fn"]:
                    parsed[f"{n}_{suffix}"] = int(row[f"{n}_{suffix}"])
                for suffix in ["gt_ratio", "pred_ratio"]:
                    parsed[f"{n}_{suffix}"] = float(row[f"{n}_{suffix}"])
                for suffix in ["gt_pixels", "pred_pixels"]:
                    parsed[f"{n}_{suffix}"] = int(row[f"{n}_{suffix}"])
            parsed["mIoU"] = float(row["mIoU"])
            parsed["Acc"] = float(row["Acc"])
            rows.append(parsed)
    return rows


# ── Compute global summary ────────────────────────────────────────────────────

def compute_summary(rows):
    """Compute per-attribute global detection metrics."""
    summary = {}
    total_images = len(rows)
    all_miou = 0.0
    all_acc = 0.0

    for r in rows:
        all_miou += r["mIoU"]
        all_acc += r["Acc"]

    for n in SEG_CLASS_NAMES:
        tp = sum(r[f"{n}_tp"] for r in rows)
        fp = sum(r[f"{n}_fp"] for r in rows)
        fn = sum(r[f"{n}_fn"] for r in rows)
        gt_present   = sum(r[f"{n}_gt_present"] for r in rows)
        pred_present = sum(r[f"{n}_pred_present"] for r in rows)
        gt_ratio_sum  = sum(r[f"{n}_gt_ratio"] for r in rows)
        pred_ratio_sum = sum(r[f"{n}_pred_ratio"] for r in rows)

        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 2 * prec * rec / max(1e-8, prec + rec)

        summary[n] = {
            "tp": tp, "fp": fp, "fn": fn,
            "gt_present": gt_present, "pred_present": pred_present,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "gt_ratio_avg": round(gt_ratio_sum / total_images, 6),
            "pred_ratio_avg": round(pred_ratio_sum / total_images, 6),
        }

    return {
        "per_class": summary,
        "global": {
            "num_images": total_images,
            "mIoU_avg": round(all_miou / total_images, 4),
            "Acc_avg": round(all_acc / total_images, 4),
        }
    }


# ── Bucket analysis ───────────────────────────────────────────────────────────

def compute_buckets(rows, attr, ratio_key, num_buckets=NUM_BUCKETS):
    """Bucket images by ratio_key, compute per-bucket detection precision."""
    data = [(r[f"{attr}_{ratio_key}"], r[f"{attr}_tp"], r[f"{attr}_pred_present"])
            for r in rows]
    data.sort(key=lambda x: x[0])

    ratios = [d[0] for d in data]
    min_r = ratios[0]
    max_r = ratios[-1]

    if max_r - min_r < 1e-9:
        total_tp = sum(d[1] for d in data)
        total_pp = sum(d[2] for d in data)
        prec = total_tp / max(1, total_pp)
        return [{"ratio_mid": round(min_r, 6), "precision": round(prec, 4),
                 "count": len(data),
                 "total_tp": total_tp, "total_pred_present": total_pp}]

    bin_width = (max_r - min_r) / num_buckets
    buckets = []

    for b in range(num_buckets):
        lo = min_r + b * bin_width
        hi = lo + bin_width
        if b == num_buckets - 1:
            hi = max_r + 1e-9

        bin_data = [d for d in data if lo <= d[0] < hi]
        if not bin_data:
            continue

        total_tp = sum(d[1] for d in bin_data)
        total_pp = sum(d[2] for d in bin_data)
        prec = total_tp / max(1, total_pp)

        buckets.append({
            "ratio_mid": round((lo + hi) / 2, 6),
            "ratio_lo": round(lo, 6),
            "ratio_hi": round(hi, 6),
            "precision": round(prec, 4),
            "count": len(bin_data),
            "total_tp": total_tp,
            "total_pred_present": total_pp,
        })

    return buckets


def compute_all_buckets(rows):
    """Compute GT-ratio and Pred-ratio bucket stats for all attributes (Precision)."""
    result = {}
    for attr in SEG_CLASS_NAMES:
        result[attr] = {
            "gt":   compute_buckets(rows, attr, "gt_ratio"),
            "pred": compute_buckets(rows, attr, "pred_ratio"),
        }
    return result


def compute_recall_buckets(rows, attr, ratio_key, num_buckets=NUM_BUCKETS):
    """Bucket images by ratio_key, compute per-bucket detection Recall.

    Recall = TP / (TP + FN) = TP / GT_present_count
    """
    data = [(r[f"{attr}_{ratio_key}"], r[f"{attr}_tp"], r[f"{attr}_gt_present"])
            for r in rows]
    data.sort(key=lambda x: x[0])

    ratios = [d[0] for d in data]
    min_r = ratios[0]
    max_r = ratios[-1]

    if max_r - min_r < 1e-9:
        total_tp = sum(d[1] for d in data)
        total_gt = sum(d[2] for d in data)
        rec = total_tp / max(1, total_gt)
        return [{"ratio_mid": round(min_r, 6), "recall": round(rec, 4),
                 "count": len(data),
                 "total_tp": total_tp, "total_gt_present": total_gt}]

    bin_width = (max_r - min_r) / num_buckets
    buckets = []

    for b in range(num_buckets):
        lo = min_r + b * bin_width
        hi = lo + bin_width
        if b == num_buckets - 1:
            hi = max_r + 1e-9

        bin_data = [d for d in data if lo <= d[0] < hi]
        if not bin_data:
            continue

        total_tp = sum(d[1] for d in bin_data)
        total_gt = sum(d[2] for d in bin_data)
        rec = total_tp / max(1, total_gt)

        buckets.append({
            "ratio_mid": round((lo + hi) / 2, 6),
            "ratio_lo": round(lo, 6),
            "ratio_hi": round(hi, 6),
            "recall": round(rec, 4),
            "count": len(bin_data),
            "total_tp": total_tp,
            "total_gt_present": total_gt,
        })

    return buckets


def compute_all_recall_buckets(rows):
    """Compute Recall bucket stats for all attributes."""
    result = {}
    for attr in SEG_CLASS_NAMES:
        result[attr] = {
            "gt":   compute_recall_buckets(rows, attr, "gt_ratio"),
            "pred": compute_recall_buckets(rows, attr, "pred_ratio"),
        }
    return result


def compute_abs_pixel_buckets(rows, attr, pixel_key, num_buckets=NUM_BUCKETS):
    """Bucket images by absolute pixel count, compute per-bucket Precision."""
    data = [(r[f"{attr}_{pixel_key}"], r[f"{attr}_tp"], r[f"{attr}_pred_present"])
            for r in rows]
    data.sort(key=lambda x: x[0])

    pixels = [d[0] for d in data]
    min_p = pixels[0]
    max_p = pixels[-1]

    if max_p - min_p < 1:
        total_tp = sum(d[1] for d in data)
        total_pp = sum(d[2] for d in data)
        prec = total_tp / max(1, total_pp)
        return [{"pixel_mid": int(min_p), "precision": round(prec, 4),
                 "count": len(data), "total_tp": total_tp,
                 "total_pred_present": total_pp}]

    bin_width = (max_p - min_p) / num_buckets
    buckets = []
    for b in range(num_buckets):
        lo = min_p + b * bin_width
        hi = lo + bin_width
        if b == num_buckets - 1:
            hi = max_p + 1
        bin_data = [d for d in data if lo <= d[0] < hi]
        if not bin_data:
            continue
        total_tp = sum(d[1] for d in bin_data)
        total_pp = sum(d[2] for d in bin_data)
        prec = total_tp / max(1, total_pp)
        buckets.append({
            "pixel_mid": int((lo + hi) / 2),
            "pixel_lo": int(lo), "pixel_hi": int(hi),
            "precision": round(prec, 4), "count": len(bin_data),
            "total_tp": total_tp, "total_pred_present": total_pp,
        })
    return buckets


def compute_all_abs_pixel_buckets(rows):
    result = {}
    for attr in SEG_CLASS_NAMES:
        result[attr] = {
            "gt":   compute_abs_pixel_buckets(rows, attr, "gt_pixels"),
            "pred": compute_abs_pixel_buckets(rows, attr, "pred_pixels"),
        }
    return result


# ── Matplotlib chart generation ───────────────────────────────────────────────

def make_bucket_chart(attr, buckets):
    """Generate a Ratio → Precision chart PNG (GT + Pred curves)."""
    gt_buckets = buckets[attr]["gt"]
    pred_buckets = buckets[attr]["pred"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if gt_buckets:
        xs = [b["ratio_mid"] for b in gt_buckets]
        ys = [b["precision"] for b in gt_buckets]
        counts = [b["count"] for b in gt_buckets]
        ax.plot(xs, ys, "o-", color="#3399FF", linewidth=2, markersize=6,
                markerfacecolor="white", markeredgewidth=2,
                label="GT Ratio → Precision", zorder=5)
        for x, y, cnt in zip(xs, ys, counts):
            ax.annotate(str(cnt), (x, y), textcoords="offset points",
                       xytext=(0, 12), fontsize=6.5, ha="center", color="#3399FF")

    if pred_buckets:
        xs = [b["ratio_mid"] for b in pred_buckets]
        ys = [b["precision"] for b in pred_buckets]
        counts = [b["count"] for b in pred_buckets]
        ax.plot(xs, ys, "s-", color="#FF8833", linewidth=2.5, markersize=6,
                markerfacecolor="white", markeredgewidth=2.5,
                label="Pred Ratio → Precision", zorder=4)
        for x, y, cnt in zip(xs, ys, counts):
            ax.annotate(str(cnt), (x, y), textcoords="offset points",
                       xytext=(0, -14), fontsize=6.5, ha="center", color="#FF8833")

    ax.set_title(DISPLAY_NAMES[attr], fontweight="bold")
    ax.set_xlabel("Pixel Ratio")
    ax.set_ylabel("Detection Precision")
    ax.set_xlim(0, None)
    ax.set_ylim(-0.02, 1.07)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def make_recall_bucket_chart(attr, buckets):
    """Generate a Ratio → Recall chart PNG (GT + Pred curves)."""
    gt_buckets = buckets[attr]["gt"]
    pred_buckets = buckets[attr]["pred"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if gt_buckets:
        xs = [b["ratio_mid"] for b in gt_buckets]
        ys = [b["recall"] for b in gt_buckets]
        counts = [b["count"] for b in gt_buckets]
        ax.plot(xs, ys, "o-", color="#2E8B57", linewidth=2, markersize=6,
                markerfacecolor="white", markeredgewidth=2,
                label="GT Ratio → Recall", zorder=5)
        for x, y, cnt in zip(xs, ys, counts):
            ax.annotate(str(cnt), (x, y), textcoords="offset points",
                       xytext=(0, 12), fontsize=6.5, ha="center", color="#2E8B57")

    if pred_buckets:
        xs = [b["ratio_mid"] for b in pred_buckets]
        ys = [b["recall"] for b in pred_buckets]
        counts = [b["count"] for b in pred_buckets]
        ax.plot(xs, ys, "s-", color="#8B4789", linewidth=2.5, markersize=6,
                markerfacecolor="white", markeredgewidth=2.5,
                label="Pred Ratio → Recall", zorder=4)
        for x, y, cnt in zip(xs, ys, counts):
            ax.annotate(str(cnt), (x, y), textcoords="offset points",
                       xytext=(0, -14), fontsize=6.5, ha="center", color="#8B4789")

    ax.set_title(DISPLAY_NAMES[attr], fontweight="bold")
    ax.set_xlabel("Pixel Ratio")
    ax.set_ylabel("Detection Recall")
    ax.set_xlim(0, None)
    ax.set_ylim(-0.02, 1.07)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def make_abs_pixel_chart(attr, buckets):
    """Generate an Absolute-Pixels → Precision chart PNG (GT + Pred curves)."""
    gt_buckets = buckets[attr]["gt"]
    pred_buckets = buckets[attr]["pred"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if gt_buckets:
        xs = [b["pixel_mid"] for b in gt_buckets]
        ys = [b["precision"] for b in gt_buckets]
        counts = [b["count"] for b in gt_buckets]
        ax.plot(xs, ys, "o-", color="#3399FF", linewidth=2, markersize=6,
                markerfacecolor="white", markeredgewidth=2,
                label="GT Pixels → Precision", zorder=5)
        for x, y, cnt in zip(xs, ys, counts):
            ax.annotate(str(cnt), (x, y), textcoords="offset points",
                       xytext=(0, 12), fontsize=6.5, ha="center", color="#3399FF")

    if pred_buckets:
        xs = [b["pixel_mid"] for b in pred_buckets]
        ys = [b["precision"] for b in pred_buckets]
        counts = [b["count"] for b in pred_buckets]
        ax.plot(xs, ys, "s-", color="#FF8833", linewidth=2.5, markersize=6,
                markerfacecolor="white", markeredgewidth=2.5,
                label="Pred Pixels → Precision", zorder=4)
        for x, y, cnt in zip(xs, ys, counts):
            ax.annotate(str(cnt), (x, y), textcoords="offset points",
                       xytext=(0, -14), fontsize=6.5, ha="center", color="#FF8833")

    ax.set_title(DISPLAY_NAMES[attr], fontweight="bold")
    ax.set_xlabel("Absolute Pixel Count")
    ax.set_ylabel("Detection Precision")
    ax.set_xlim(0, None)
    ax.set_ylim(-0.02, 1.07)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}'))
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ── Trimmed bucket analysis (95th percentile, finer granularity) ─────────────

def compute_trimmed_buckets(rows, attr, ratio_key, pct=95, num_buckets=20):
    """Bucket by ratio_key using only the smallest pct% of non-zero values."""
    vals = [r[f"{attr}_{ratio_key}"] for r in rows if r[f"{attr}_{ratio_key}"] > 0]
    if not vals:
        return []
    vals.sort()
    cutoff = vals[int(len(vals) * pct / 100)]

    data = [(r[f"{attr}_{ratio_key}"], r[f"{attr}_tp"], r[f"{attr}_pred_present"])
            for r in rows if 0 < r[f"{attr}_{ratio_key}"] <= cutoff]
    data.sort(key=lambda x: x[0])

    if len(data) < num_buckets:
        num_buckets = max(2, len(data) // 50)

    min_r, max_r = data[0][0], data[-1][0]
    if max_r - min_r < 1e-9:
        total_tp = sum(d[1] for d in data)
        total_pp = sum(d[2] for d in data)
        prec = total_tp / max(1, total_pp)
        return [{"ratio_mid": round(min_r, 6), "precision": round(prec, 4),
                 "count": len(data), "total_tp": total_tp, "total_pred_present": total_pp,
                 "ratio_lo": round(min_r, 6), "ratio_hi": round(max_r, 6)}]

    bin_width = (max_r - min_r) / num_buckets
    buckets = []
    for b in range(num_buckets):
        lo = min_r + b * bin_width
        hi = lo + bin_width
        if b == num_buckets - 1:
            hi = max_r + 1e-9
        bin_data = [d for d in data if lo <= d[0] < hi]
        if not bin_data:
            continue
        total_tp = sum(d[1] for d in bin_data)
        total_pp = sum(d[2] for d in bin_data)
        prec = total_tp / max(1, total_pp)
        buckets.append({
            "ratio_mid": round((lo + hi) / 2, 6),
            "ratio_lo": round(lo, 6), "ratio_hi": round(hi, 6),
            "precision": round(prec, 4), "count": len(bin_data),
            "total_tp": total_tp, "total_pred_present": total_pp,
        })
    return buckets


def make_trimmed_ratio_chart(attr, gt_buckets, pred_buckets):
    """Generate a trimmed Ratio → Precision chart (95th pct, 20 bins)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if gt_buckets:
        xs = [b["ratio_mid"] for b in gt_buckets]
        ys = [b["precision"] for b in gt_buckets]
        counts = [b["count"] for b in gt_buckets]
        ax.plot(xs, ys, "o-", color="#3399FF", linewidth=2, markersize=5,
                markerfacecolor="white", markeredgewidth=1.5,
                label="GT Ratio → Precision", zorder=5)
        for i, (x, y, cnt) in enumerate(zip(xs, ys, counts)):
            if i % 3 == 0:  # sparser labels for 20 buckets
                ax.annotate(str(cnt), (x, y), textcoords="offset points",
                           xytext=(0, 10), fontsize=6, ha="center", color="#3399FF")

    if pred_buckets:
        xs = [b["ratio_mid"] for b in pred_buckets]
        ys = [b["precision"] for b in pred_buckets]
        counts = [b["count"] for b in pred_buckets]
        ax.plot(xs, ys, "s-", color="#FF8833", linewidth=2.5, markersize=5,
                markerfacecolor="white", markeredgewidth=1.5,
                label="Pred Ratio → Precision", zorder=4)
        for i, (x, y, cnt) in enumerate(zip(xs, ys, counts)):
            if i % 3 == 0:
                ax.annotate(str(cnt), (x, y), textcoords="offset points",
                           xytext=(0, -12), fontsize=6, ha="center", color="#FF8833")

    ax.set_title(DISPLAY_NAMES[attr], fontweight="bold")
    ax.set_xlabel("Pixel Ratio (trimmed 95th pct)")
    ax.set_ylabel("Detection Precision")
    ax.set_xlim(0, None)
    ax.set_ylim(-0.02, 1.07)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def compute_trimmed_abs_buckets(rows, attr, pixel_key, pct=95, num_buckets=20):
    """Bucket by absolute pixel count using only the smallest pct% of non-zero values."""
    vals = [r[f"{attr}_{pixel_key}"] for r in rows if r[f"{attr}_{pixel_key}"] > 0]
    if not vals:
        return []
    vals.sort()
    cutoff = vals[int(len(vals) * pct / 100)]

    data = [(r[f"{attr}_{pixel_key}"], r[f"{attr}_tp"], r[f"{attr}_pred_present"])
            for r in rows if 0 < r[f"{attr}_{pixel_key}"] <= cutoff]
    data.sort(key=lambda x: x[0])

    if len(data) < num_buckets:
        num_buckets = max(2, len(data) // 50)

    min_p, max_p = data[0][0], data[-1][0]
    if max_p - min_p < 1:
        total_tp = sum(d[1] for d in data)
        total_pp = sum(d[2] for d in data)
        prec = total_tp / max(1, total_pp)
        return [{"pixel_mid": int(min_p), "precision": round(prec, 4),
                 "count": len(data), "total_tp": total_tp, "total_pred_present": total_pp}]

    bin_width = (max_p - min_p) / num_buckets
    buckets = []
    for b in range(num_buckets):
        lo = min_p + b * bin_width
        hi = lo + bin_width
        if b == num_buckets - 1:
            hi = max_p + 1
        bin_data = [d for d in data if lo <= d[0] < hi]
        if not bin_data:
            continue
        total_tp = sum(d[1] for d in bin_data)
        total_pp = sum(d[2] for d in bin_data)
        prec = total_tp / max(1, total_pp)
        buckets.append({
            "pixel_mid": int((lo + hi) / 2),
            "precision": round(prec, 4), "count": len(bin_data),
            "total_tp": total_tp, "total_pred_present": total_pp,
        })
    return buckets


def make_trimmed_abs_chart(attr, gt_buckets, pred_buckets):
    """Generate a trimmed Absolute-Pixel → Precision chart (95th pct, 20 bins)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if gt_buckets:
        xs = [b["pixel_mid"] for b in gt_buckets]
        ys = [b["precision"] for b in gt_buckets]
        counts = [b["count"] for b in gt_buckets]
        ax.plot(xs, ys, "o-", color="#3399FF", linewidth=2, markersize=5,
                markerfacecolor="white", markeredgewidth=1.5,
                label="GT Pixels → Precision", zorder=5)
        for i, (x, y, cnt) in enumerate(zip(xs, ys, counts)):
            if i % 3 == 0:
                ax.annotate(str(cnt), (x, y), textcoords="offset points",
                           xytext=(0, 10), fontsize=6, ha="center", color="#3399FF")

    if pred_buckets:
        xs = [b["pixel_mid"] for b in pred_buckets]
        ys = [b["precision"] for b in pred_buckets]
        counts = [b["count"] for b in pred_buckets]
        ax.plot(xs, ys, "s-", color="#FF8833", linewidth=2.5, markersize=5,
                markerfacecolor="white", markeredgewidth=1.5,
                label="Pred Pixels → Precision", zorder=4)
        for i, (x, y, cnt) in enumerate(zip(xs, ys, counts)):
            if i % 3 == 0:
                ax.annotate(str(cnt), (x, y), textcoords="offset points",
                           xytext=(0, -12), fontsize=6, ha="center", color="#FF8833")

    ax.set_title(DISPLAY_NAMES[attr], fontweight="bold")
    ax.set_xlabel("Absolute Pixel Count (trimmed 95th pct)")
    ax.set_ylabel("Detection Precision")
    ax.set_xlim(0, None)
    ax.set_ylim(-0.02, 1.07)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}'))
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ── Badcase loading ───────────────────────────────────────────────────────────

def load_badcases(badcase_dir):
    """Load badcase meta and encode images as base64 PNG."""
    meta_path = Path(badcase_dir) / "meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    masks_dir = Path(badcase_dir) / "masks"

    for img_info in meta["images"]:
        stem = img_info["stem"]

        # Read RGB
        rgb_path = masks_dir / f"{stem}_rgb.png"
        img_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

        # Read GT mask
        gt_path = masks_dir / f"{stem}_gt.npy"
        gt_mask = np.load(gt_path) if gt_path.exists() else np.zeros((256, 256), dtype=np.uint8)

        # Read Pred mask
        pred_path = masks_dir / f"{stem}_pred.npy"
        pred_mask = np.load(pred_path) if pred_path.exists() else np.zeros((256, 256), dtype=np.uint8)

        # Encode as base64 PNG
        img_info["rgb_b64"] = _ndarray_to_b64(img_rgb)
        img_info["gt_b64"]  = _ndarray_to_b64(_colorize_mask(gt_mask))
        img_info["pred_b64"] = _ndarray_to_b64(_colorize_mask(pred_mask))

        # Build FP/FN annotation per class
        fp_classes = []
        fn_classes = []
        row = img_info["row"]
        for n in SEG_CLASS_NAMES:
            if row.get(f"{n}_fp", 0) == 1:
                fp_classes.append(n)
            if row.get(f"{n}_fn", 0) == 1:
                fn_classes.append(n)
        img_info["fp_classes"] = fp_classes
        img_info["fn_classes"] = fn_classes

    return meta


def _colorize_mask(mask_hw):
    """Convert uint8 seg mask (H,W) to RGB image (H,W,3)."""
    h, w = mask_hw.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in SEG_COLORS_RGB.items():
        color[mask_hw == cls_id] = rgb
    return color


def _ndarray_to_b64(img_rgb):
    """Encode an RGB numpy array as a base64 PNG string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", img_bgr)
    if not success:
        return ""
    return base64.b64encode(buf).decode("ascii")


# ── HTML generation ───────────────────────────────────────────────────────────

def generate_html(summary, chart_images, recall_chart_images, abs_chart_images,
                  trimmed_chart_images, trimmed_abs_chart_images, badcases, output_path):
    """Generate self-contained HTML report with matplotlib chart images."""

    # ── Summary table rows ──
    table_rows = ""
    for n in SEG_CLASS_NAMES:
        s = summary["per_class"][n]

        # Highlight best/worst
        table_rows += f"""
            <tr>
                <td class="class-label">{DISPLAY_NAMES[n]}</td>
                <td>{s['precision']:.4f}</td>
                <td>{s['recall']:.4f}</td>
                <td>{s['f1']:.4f}</td>
                <td>{s['pred_ratio_avg']:.6f}</td>
                <td>{s['pred_present']}</td>
                <td>{s['tp']}</td>
                <td>{s['fp']}</td>
                <td>{s['fn']}</td>
            </tr>"""

    # ── Precision chart images ──
    chart_html = ""
    for i, n in enumerate(SEG_CLASS_NAMES):
        b64 = chart_images.get(n, "")
        chart_html += f"""
        <div class="chart-card">
            <img src="data:image/png;base64,{b64}" alt="{n} precision chart" style="width:100%;height:auto;">
        </div>"""

    # ── Recall chart images ──
    recall_chart_html = ""
    for i, n in enumerate(SEG_CLASS_NAMES):
        b64 = recall_chart_images.get(n, "")
        recall_chart_html += f"""
        <div class="chart-card">
            <img src="data:image/png;base64,{b64}" alt="{n} recall chart" style="width:100%;height:auto;">
        </div>"""

    # ── Absolute pixel Precision chart images ──
    abs_chart_html = ""
    for i, n in enumerate(SEG_CLASS_NAMES):
        b64 = abs_chart_images.get(n, "")
        abs_chart_html += f"""
        <div class="chart-card">
            <img src="data:image/png;base64,{b64}" alt="{n} abs pixel chart" style="width:100%;height:auto;">
        </div>"""

    # ── Dataset detection ──
    is_train = summary['global']['num_images'] > 50000

    # ── Trimmed ratio chart images ──
    trimmed_chart_html = ""
    for i, n in enumerate(SEG_CLASS_NAMES):
        b64 = trimmed_chart_images.get(n, "")
        trimmed_chart_html += f"""
        <div class="chart-card">
            <img src="data:image/png;base64,{b64}" alt="{n} trimmed ratio chart" style="width:100%;height:auto;">
        </div>"""

    # ── Trimmed absolute pixel chart images ──
    trimmed_abs_chart_html = ""
    for i, n in enumerate(SEG_CLASS_NAMES):
        b64 = trimmed_abs_chart_images.get(n, "")
        trimmed_abs_chart_html += f"""
        <div class="chart-card">
            <img src="data:image/png;base64,{b64}" alt="{n} trimmed abs chart" style="width:100%;height:auto;">
        </div>"""

    # ── Badcase ──
    badcase_html = _build_badcase_html(badcases)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Detection Precision vs Pixel Ratio — iter_22000</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5; color: #333; line-height: 1.6;
    padding: 20px;
}}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ text-align: center; margin-bottom: 8px; font-size: 24px; color: #1a1a1a; }}
.subtitle {{ text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }}

/* ── Section titles ── */
.section-title {{
    font-size: 20px; font-weight: 700; margin: 30px 0 15px;
    padding-bottom: 8px; border-bottom: 3px solid #4A90D9; color: #1a1a1a;
}}

/* ── Summary Table ── */
.summary-box {{
    background: #fff; border-radius: 12px; padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px;
    overflow-x: auto;
}}
table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
}}
th {{ background: #4A90D9; color: #fff; padding: 10px 8px; text-align: center;
      font-weight: 600; }}
td {{ padding: 8px; text-align: center; border-bottom: 1px solid #eee; }}
tr:hover td {{ background: #f0f6ff; }}
.class-label {{ font-weight: 600; text-align: left; padding-left: 12px; }}

/* ── Charts Grid ── */
.charts-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    margin-bottom: 20px;
}}
@media (max-width: 1200px) {{ .charts-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
@media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
@media (max-width: 600px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
.chart-card {{
    background: #fff; border-radius: 12px; padding: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
}}
.chart-card img {{
    max-width: 100%; height: auto; border-radius: 6px;
}}

/* ── Badcase ── */
.badcase-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 16px; margin-top: 15px;
}}
.badcase-card {{
    background: #fff; border-radius: 12px; padding: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    font-size: 12px;
}}
.badcase-card .img-row {{
    display: flex; gap: 8px; margin-bottom: 8px;
}}
.badcase-card .img-row img {{
    width: 32%; height: auto; border-radius: 6px; border: 1px solid #ddd;
}}
.badcase-card .info {{ margin-bottom: 4px; }}
.badcase-card .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 700; margin-right: 4px;
}}
.badge-fp {{ background: #ffebee; color: #c62828; }}
.badge-fn {{ background: #fff3e0; color: #e65100; }}
.badge-both {{ background: #fce4ec; color: #880e4f; }}
.badcase-card .tag {{
    display: inline-block; background: #f0f0f0; padding: 1px 6px;
    border-radius: 3px; margin: 1px 2px; font-size: 11px;
}}

/* ── Analysis section ── */
.analysis-box {{
    background: #fff; border-radius: 12px; padding: 24px 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px;
    font-size: 14px; line-height: 1.8;
}}
.analysis-box h3 {{
    font-size: 16px; color: #1a1a1a; margin: 20px 0 10px;
    padding-bottom: 6px; border-bottom: 1px solid #e0e0e0;
}}
.analysis-box h3:first-child {{ margin-top: 0; }}
.analysis-box p {{ margin: 8px 0; color: #444; }}
.analysis-box ul {{ margin: 6px 0 6px 20px; color: #444; }}
.analysis-box li {{ margin: 3px 0; }}
.analysis-box .highlight {{
    background: #fff3cd; border-left: 4px solid #ffc107;
    padding: 10px 14px; margin: 12px 0; border-radius: 4px;
    font-weight: 500;
}}
.analysis-box .insight {{
    background: #e8f5e9; border-left: 4px solid #4caf50;
    padding: 10px 14px; margin: 12px 0; border-radius: 4px;
}}
.analysis-box .warn {{
    background: #ffebee; border-left: 4px solid #e53935;
    padding: 10px 14px; margin: 12px 0; border-radius: 4px;
}}
.analysis-box .mini-table {{
    width: auto; font-size: 12px; margin: 8px 0;
}}
.analysis-box .mini-table th {{
    font-size: 11px; padding: 6px 10px;
}}
.analysis-box .mini-table td {{
    font-size: 12px; padding: 5px 10px;
}}
</style>
</head>
<body>
<div class="container">

<h1>人体属性像素占比 vs 检出 Precision 关系分析</h1>
<p class="subtitle">
    Checkpoint: iter_22000 (RGB-only, OptC) &nbsp;|&nbsp;
    Val set: {summary['global']['num_images']} images &nbsp;|&nbsp;
    Avg mIoU: {summary['global']['mIoU_avg']:.4f} &nbsp;|&nbsp;
    Avg Acc: {summary['global']['Acc_avg']:.4f}
</p>

<!-- ── Summary Table ── -->
<h2 class="section-title">一、全局检出指标汇总</h2>
<div class="summary-box">
    <table>
        <thead>
            <tr>
                <th>属性</th>
                <th>Precision</th><th>Recall</th><th>F1</th>
                <th>Pred Avg Ratio</th>
                <th>Pred Present</th>
                <th>TP</th><th>FP</th><th>FN</th>
            </tr>
        </thead>
        <tbody>{table_rows}
        </tbody>
    </table>
</div>

<!-- ── Section 二: Ratio Precision Charts + Inflection Analysis ── -->
<h2 class="section-title">二、像素占比 vs 检出 Precision 分桶曲线</h2>
<p style="color:#666;font-size:13px;margin-bottom:10px;">
    每个属性分 10 个等宽桶。蓝线 = GT像素占比→Precision；橙线 = Pred像素占比→Precision。
    散点旁标注桶内样本数。
</p>
<div class="charts-grid">{chart_html}
</div>

{_build_ratio_inflection_html(summary, is_train)}

<!-- ── Section 三: Absolute Pixel Precision Charts + Inflection Analysis ── -->
<h2 class="section-title">三、绝对像素数 vs 检出 Precision 分桶曲线</h2>
<p style="color:#666;font-size:13px;margin-bottom:10px;">
    每个属性分 10 个等宽桶。蓝线 = GT绝对像素→Precision；橙线 = Pred绝对像素→Precision。
    X 轴为绝对像素数，更直观对应部署场景。
</p>
<div class="charts-grid">{abs_chart_html}
</div>

{_build_abs_inflection_html(summary, is_train)}

<!-- ── Section 四: Recall Charts + Recall Analysis ── -->
<h2 class="section-title">四、像素占比 vs 检出 Recall 分桶曲线</h2>
<p style="color:#666;font-size:13px;margin-bottom:10px;">
    每个属性分 10 个等宽桶。绿线 = GT像素占比→Recall；紫线 = Pred像素占比→Recall。
    散点旁标注桶内样本数。
</p>
<div class="charts-grid">{recall_chart_html}
</div>

{_build_full_analysis_html(summary, is_train)}

<!-- ── Badcase ── -->
{badcase_html}

<!-- ── Trimmed Charts ── -->
<h2 class="section-title">六、像素占比 vs 检出 Precision（去除 top 5% 离群值，20 桶细粒度）</h2>
<p style="color:#666;font-size:13px;margin-bottom:10px;">
    只保留 GT/Pred ratio 最小的 95% 数据，分 20 个等宽桶。蓝线 = GT像素占比→Precision；橙线 = Pred像素占比→Precision。
</p>
<div class="charts-grid">{trimmed_chart_html}
</div>

<h2 class="section-title">七、绝对像素数 vs 检出 Precision（去除 top 5% 离群值，20 桶细粒度）</h2>
<p style="color:#666;font-size:13px;margin-bottom:10px;">
    只保留 GT/Pred 绝对像素数最小的 95% 数据，分 20 个等宽桶。蓝线 = GT像素数→Precision；橙线 = Pred像素数→Precision。
</p>
<div class="charts-grid">{trimmed_abs_chart_html}
</div>

</div><!-- .container -->
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _build_full_analysis_html(summary, is_train=False):
    """Full analysis section: overview + Precision/Recall findings + suggestions."""
    s = summary["per_class"]
    g = summary["global"]

    diag_rows = ""
    for n in SEG_CLASS_NAMES:
        prec = s[n]["precision"]
        rec  = s[n]["recall"]
        f1   = s[n]["f1"]
        issues = []
        if prec < 0.75: issues.append("FP严重")
        elif prec < 0.85: issues.append("FP较多")
        if rec < 0.50: issues.append("FN严重(大量漏检)")
        elif rec < 0.80: issues.append("FN较多(漏检)")
        if prec >= 0.90 and rec >= 0.90: issues.append("良好")
        elif prec >= 0.85 and rec >= 0.80: issues.append("一般")
        diag = "，".join(issues)
        diag_rows += f"""<tr><td class="class-label">{DISPLAY_NAMES[n]}</td>
            <td>{prec:.4f}</td><td>{rec:.4f}</td><td>{f1:.4f}</td><td>{diag}</td></tr>"""

    if is_train:
        return f"""
<div class="analysis-box">

<h3>训练集整体表现概览</h3>
<table class="mini-table">
    <thead><tr><th>属性</th><th>Precision</th><th>Recall</th><th>F1</th><th>诊断</th></tr></thead>
    <tbody>{diag_rows}</tbody>
</table>
<div class="warn">
    <b>注意</b>：这是在 <b>训练集</b>（{g['num_images']} 张图）上的推理结果。由于模型使用 iter_22000 中间检查点（训练未完全收敛），
    训练集上的表现不代表最终模型能力，但可以揭示<b>模型在训练过程中的行为特征和偏差倾向</b>。
</div>

<h3>训练集核心发现</h3>

<h4>① Mask — 灾难性 FP，模型在训练集上严重「幻觉」口罩</h4>
<p>
    Mask 的 Precision 仅 <b>0.144</b>——模型预测了 9091 张图有口罩，但实际只有 2117 张有。
    <b>7778 个 FP vs 1313 个 TP</b>，意味着模型在训练集上大量「幻想」口罩的存在。
    这很可能是训练过程中的阶段性现象：模型先学会「生成」口罩区域（大量 FP），随后才学会精准控制（降低 FP）。
    iter_22000 时模型尚处于 FP 高发阶段。
</p>
<div class="highlight">
    <b>部署启示</b>：如果在训练集上 Mask 的 FP 率都如此之高，说明模型对口罩的判别能力尚未建立。
    在验证集上 Mask P=0.818 说明模型确实在进步。部署时对 Mask 属性需要设置较高的 Pred 像素阈值（如 ≥100 pixels），
    以过滤掉低面积的虚假口罩检出。
</div>

<h4>② Glasses — 大量漏检（FN=8121），模型对眼镜「视而不见」</h4>
<p>
    Glasses 的 Recall 仅 <b>0.428</b>——14184 张有眼镜的图中，模型漏掉了 8121 张。
    同时 Precision=0.724，说明模型预测的眼镜也有 28% 是错的。
    <b>既漏检又多误检</b>，说明眼镜这个类别对模型来说难度很大。
    从 Pred 占比分桶看，绝大多数图像的 Pred ratio=0（模型完全没有预测眼镜），仅在最高 20% 的桶中才有非零预测。
</p>

<h4>③ Hat — 近 5000 个 FP，帽子是第二大误检源</h4>
<p>
    Hat 的 FP 高达 <b>4835</b>（Precision=0.714），FN 也有 4895（Recall=0.711）。
    模型对帽子的预测处于「乱猜」阶段——几乎等量的 FP 和 FN。
    当 Pred ratio &gt; 0.8% 时 Precision 才爬升到 0.81，Pred ratio &lt; 0.8% 时 Precision 仅 0.45。
</p>

<h4>④ Hair — 唯一稳健的属性</h4>
<p>
    Hair 的 Precision=0.931、Recall=0.914，是四个头部属性中唯一表现良好的。
    当 Pred ratio &gt; 1.4% 时 Precision 升至 0.96+；当 Pred pixels &gt; 259 时 Precision 达 0.94+。
    Hair 在训练集上的表现与验证集接近，说明模型对头发的分割能力已经基本建立。
</p>

<h3>训练集 vs 验证集对比</h3>
<table class="mini-table">
    <thead><tr><th>属性</th><th>训练集 P</th><th>验证集 P</th><th>训练集 R</th><th>验证集 R</th><th>变化</th></tr></thead>
    <tbody>
        <tr><td>hair</td><td>0.931</td><td>0.916</td><td>0.914</td><td>0.932</td><td>基本一致</td></tr>
        <tr><td>hat</td><td>0.714</td><td>0.856</td><td>0.711</td><td>0.772</td><td>训练集显著更差（FP多）</td></tr>
        <tr style="background:#ffcdd2;"><td>glasses</td><td>0.724</td><td>0.816</td><td>0.428</td><td>0.609</td><td>训练集大幅更差（FN多）</td></tr>
        <tr style="background:#ffcdd2;"><td>mask</td><td>0.144</td><td>0.818</td><td>0.620</td><td>0.574</td><td>训练集 Precision 崩溃（FP爆炸）</td></tr>
    </tbody>
</table>
<p>
    <b>Hair</b> 训练/验证表现接近 → 模型对头发的学习已经稳定。<br>
    <b>Hat/Glasses/Mask</b> 训练集远差于验证集 → 模型在训练过程中仍在学习这些小属性，尚未收敛。
    验证集上的提升说明模型的泛化方向是正确的，但训练过程中的大量 FP 说明这些小类需要更多迭代或更强的监督信号。
</p>

<h3>部署建议（基于训练集分析的补充）</h3>
<ul>
    <li><b>Hair</b>：Pred pixels ≥ 260 可作为可靠阈值（Precision 达 0.94+）。</li>
    <li><b>Hat</b>：训练集 FP 极高，部署时建议 Pred pixels ≥ 160（Precision 约 0.80）。</li>
    <li><b>Glasses</b>：Recall 严重不足（42%），需要从模型层面提升，单纯依赖 Pred 阈值效果有限。</li>
    <li><b>Mask</b>：当前检查点在训练集上的 Mask 预测基本不可用（P=0.14）。应使用更晚的检查点（如 iter_33000）再评估。</li>
</ul>

</div>"""
    else:
        return f"""
<div class="analysis-box">

<h3>整体表现概览</h3>
<table class="mini-table">
    <thead><tr><th>属性</th><th>Precision</th><th>Recall</th><th>F1</th><th>诊断</th></tr></thead>
    <tbody>{diag_rows}</tbody>
</table>

<h3>Precision 分析：Pred 像素占比越大，误检越少</h3>
<p>
    四个头部属性中，当模型<b>预测的像素占比（Pred Ratio）增大时，Precision 单调上升</b>——模型对大面积预测更有信心。
    上方曲线图展示了每个属性 Pred Ratio 分桶后的 Precision 变化，可以直观看出：预测面积越大，检出越可靠。
</p>
<ul>
    <li><b>Hair</b>：Pred 占比 > 0.3% 时 Precision 稳定在 0.95+。FP 主要来自极小面积的误报。</li>
    <li><b>Hat</b>：Pred 占比 > 1~2% 时 Precision 接近 1.0。极低占比的「帽子」检出大概率是误报。</li>
    <li><b>Glasses</b>：Pred 占比 > 0.1% 时 Precision 跃升至 0.90+。极低占比时模型容易把小区域噪声误判为眼镜。</li>
    <li><b>Mask</b>：样本最少，Precision 在低 Pred 占比时波动较大，但占比越大 Precision 越高的趋势成立。</li>
</ul>

<h3>Recall 分析：Pred 像素占比越大，漏检越少</h3>
<ul>
    <li><b>Glasses / Mask</b>：Recall 随 Pred 占比急速上升。</li>
    <li><b>Hat</b>：Pred 占比 < 1~2% 时 Recall 偏低，> 2% 后跃升。</li>
    <li><b>Hair</b>：Recall 始终较高，但极低 Pred 占比时仍有少量漏检。</li>
</ul>
<div class="highlight">
    <b>核心发现</b>：四个头部属性的 Precision 和 Recall 都与<b>模型预测的像素面积</b>呈显著正相关。
    Pred 面积越大→模型越确信→结果越可靠。
</div>

<h3>GT vs Pred 曲线对比</h3>
<ul>
    <li><b>Precision 图（蓝=GT，橙=Pred）</b>：蓝线（GT占比→Precision）在 GT 占比稍大时几乎立即达到 1.0——说明只要属性真实存在且有足够面积，模型几乎不会误检。橙线（Pred占比→Precision）从低到高爬升——反映模型从不确定到确信的过程。两条线的间距越大，说明模型对低面积预测越容易 FP。</li>
    <li><b>Recall 图（绿=GT，紫=Pred）</b>：绿线（GT占比→Recall）反映「真实大小对漏检的影响」——GT 占比越大，Recall 越高。紫线（Pred占比→Recall）反映「模型预测大小对漏检的影响」。紫线在绿线下方 → 模型倾向于低估面积；紫线在绿线上方 → 模型倾向于高估面积。</li>
    <li>标注在点旁的数字是<b>该桶的图片数量</b>，样本少的桶（<50）不具参考价值。</li>
</ul>

<h3>改善建议</h3>
<table class="mini-table">
    <thead><tr><th>问题</th><th>建议</th></tr></thead>
    <tbody>
        <tr><td>小属性（Glasses/Mask）Recall 低</td><td>增加小类样本量；使用 Focal Loss 或更大 loss weight</td></tr>
        <tr><td>Hat 低面积时 Precision/Recall 骤降</td><td>增加多尺度训练；加入小目标增强</td></tr>
        <tr><td>Hair FP（与 Hat 相关）</td><td>加入 Hat/Hair 联合训练约束</td></tr>
        <tr><td>整体</td><td>检出问题本质上是分割质量问题</td></tr>
    </tbody>
</table>

</div>"""


def _build_ratio_inflection_html(summary, is_train=False):
    """Pixel ratio inflection analysis — follows ratio precision charts."""
    if is_train:
        return """
<div class="analysis-box">

<h3>训练集像素占比曲线解读</h3>

<p>
    上方 4 张曲线图展示了训练集上<b>模型预测的像素占比（Pred Ratio）</b>与检出 Precision 的分桶关系。
    曲线形态反映了模型在训练过程中的行为特征。
</p>

<h4>Hair — Pred 占比 > 1.4% 后 Precision 跃升至 0.96</h4>
<p>
    当 Pred ratio 在 0~1.4% 区间时 Precision 仅 0.78；超过 1.4% 后跃升至 0.96；超过 2.5% 后稳定在 0.98+。
    <b>1.4% 是 Hair 在训练集上的关键拐点</b>。低于此值的检出中约 22% 是 FP。
</p>

<h4>Hat — 大量零预测图像拉低了整体 Precision</h4>
<p>
    训练集中超过 60% 的图像 Pred ratio=0（模型完全未预测帽子），这些图像的 Precision 为 0（因为部分图像 GT 有帽子但 Pred=0）。
    当 Pred ratio > 0.8% 后 Precision 爬升至 0.81。说明帽子需要一定的预测面积才可信。
</p>

<h4>Glasses — 仅最高 20% 的图像有非零预测</h4>
<p>
    绝大多数训练图像的 Pred ratio=0，模型对眼镜的预测非常保守。
    在少量有预测的图像中（最高 20% 桶），Precision=0.72——即使模型主动预测了眼镜，也有 28% 是错的。
</p>

<h4>Mask — 训练阶段 FP 爆炸</h4>
<p>
    与验证集不同，训练集上 Mask 的 Precision 极低（0.14）。模型预测了大量口罩区域（9091 个 Pred present），
    但其中 86% 是 FP。这是训练过程中的阶段性现象，曲线图中 Precision 始终在低位徘徊。
</p>

</div>"""
    else:
        return """
<div class="analysis-box">

<h3>像素占比曲线关键解读</h3>
<p>
    上方 4 张曲线图展示了<b>模型预测的像素占比（Pred Ratio）</b>与检出 Precision 的分桶关系。
</p>
<h4>Hair — Pred 占比达到 0.3% 后 Precision 稳定在高位</h4>
<p>当 Pred ratio < 0.3% 时 Precision 约 0.88；超过 0.3% 后迅速攀升至 0.95+。<b>0.3% 是可靠拐点</b>。</p>
<h4>Glasses — 极低 Pred 占比下的 Precision 悬崖</h4>
<p>Pred 占比 < 0.05% 时 Precision 仅 0.6~0.7；> 0.15% 后稳定在 0.90+。</p>
<h4>Hat — Pred 占比 1~2% 是分水岭</h4>
<p>Pred 占比 < 1% 时 Precision 约 0.82，> 2% 后跃升至 0.95+。</p>
<h4>Mask — 样本少导致曲线波动</h4>
<p>Mask 样本量较少，但 Pred 占比越大 Precision 越高的趋势成立。</p>
<div class="highlight"><b>部署建议</b>：在曲线上找到 Precision 达到期望值的拐点，读出对应的 Pred 像素占比即为推荐阈值。</div>

<h4>GT vs Pred 曲线对比</h4>
<p>上方每张图有两条曲线：</p>
<ul>
    <li><b>蓝线（GT 占比→Precision）</b>：按真实标注的像素占比分桶。反映「属性真实存在且足够大时，模型表现如何」——可看作模型能力的<b>理论上限</b>。GT 占比稍大时（如 Hair >1.2%、Hat >1.4%、Glasses >0.2%），Precision 几乎立即达到 1.0，说明<b>只要属性真实且可见，模型几乎不会误检</b>。</li>
    <li><b>橙线（Pred 占比→Precision）</b>：按模型预测的像素占比分桶。反映「部署时实际可用的信号」——你只能看到 Pred。橙线从低到高的爬升过程，就是模型从「乱猜」到「确信」的过程。</li>
    <li><b>两条线的间距</b>：蓝线在上、橙线在下 → 模型对低占比预测容易 FP（预测了但实际没有）。间距越大，说明低 Pred 占比的检出越不可靠。Hair 的间距最小（模型预测面积接近真实），Mask 的间距最大。</li>
</ul>
</div>"""


def _build_abs_inflection_html(summary, is_train=False):
    """Absolute pixel inflection analysis — follows absolute pixel charts."""
    if is_train:
        return """
<div class="analysis-box">

<h3>训练集绝对像素数曲线解读</h3>

<p>
    上方 4 张曲线图展示了训练集上<b>模型预测的绝对像素数（Pred Pixels）</b>与检出 Precision 的分桶关系。
</p>

<h4>Hair — Pred pixels > 260 后 Precision 达 0.94</h4>
<p>
    训练集上 Hair 的绝对像素拐点约为 <b>260 pixels</b>：低于此值时 Precision 约 0.83，超过后跃升至 0.94。
    与验证集（~100 pixels）相比，训练集需要更高的像素阈值才能达到同等 Precision，
    这是因为训练集上模型对 Hair 的 FP 更多（3333 vs 517）。
</p>

<h4>Hat — Pred pixels > 164 后 Precision 爬升至 0.80</h4>
<p>
    训练集上 Hat 的 FP 高达 4835，导致整体 Precision 仅 0.71。
    当 Pred pixels > 164 时 Precision 升至 0.80，但仍远低于验证集同条件下的 0.90+。
</p>

<h4>Glasses — 仅少量图像有非零预测</h4>
<p>
    训练集上模型对 Glasses 的预测非常稀疏，仅有约 20% 的图像 Pred pixels > 0。
    在这些有预测的图像中，Precision 约 0.72。
</p>

<h4>Mask — 不宜参考</h4>
<p>
    训练集上 Mask 的 Precision=0.144，绝对像素曲线在低位波动，不宜作为部署参考。
    应使用验证集的结果确定 Mask 的像素阈值。
</p>

</div>"""
    else:
        return """
<div class="analysis-box">

<h3>绝对像素数曲线关键解读</h3>
<p>上方 4 张曲线图展示了<b>模型预测的绝对像素数（Pred Pixels）</b>与检出 Precision 的分桶关系。</p>
<h4>Hair — 100 pixels 是临界点</h4>
<p>Pred pixels > 100 时 Precision 稳定在 0.93+；> 500 时接近 1.0。</p>
<h4>Glasses — 需要约 50 pixels 才可靠</h4>
<p>Pred pixels < 20 时 Precision 仅 0.7；> 50~100 时上升到 0.85~0.90。</p>
<h4>Hat — 需要 300+ pixels</h4>
<p>Pred pixels > 300 时 Precision 可达 0.90+；> 1000 时稳定在 0.95+。</p>
<h4>Mask — 需要 100+ pixels</h4>
<p>Pred pixels > 100 时 Precision 达 0.90；> 300 时稳定在 0.95+。</p>
<div class="highlight"><b>部署建议</b>：在 Pred pixels 曲线上找到期望 Precision 对应的 X 轴像素数即为推荐阈值。</div>

<h4>GT vs Pred 曲线对比</h4>
<ul>
    <li><b>蓝线（GT 像素→Precision）</b>：按真实标注的绝对像素数分桶。GT 像素稍多时 Precision 迅速达到 1.0，验证了「属性真实且面积足够时模型不会误检」。</li>
    <li><b>橙线（Pred 像素→Precision）</b>：部署时可用的信号。橙色线的爬升过程即模型置信度的建立过程。</li>
    <li><b>横向间距</b>：两条线在同一 Precision 水平时 X 轴的差距，反映模型对该属性的<b>预测偏差</b>。例如 Hair 在 P=0.95 时 GT~100px 但 Pred~200px，说明模型预测的头发面积略大于真实值。</li>
</ul>
</div>"""


def _build_badcase_html(badcases):
    """Build HTML for badcase visualization section."""
    if badcases is None or len(badcases.get("images", [])) == 0:
        return """<h2 class="section-title">三、Badcase 展示</h2>
        <p style="color:#888;">No badcase data found.</p>"""

    parts = ['<h2 class="section-title">五、Badcase 展示</h2>']
    parts.append('<p style="color:#666;font-size:13px;margin-bottom:10px;">')
    parts.append('左=RGB &nbsp; 中=GT Segmentation &nbsp; 右=Pred Segmentation<br>')
    parts.append('FP（红色标签）= 模型多检的属性；FN（橙色标签）= 模型漏检的属性')
    parts.append('</p>')
    parts.append('<div class="badcase-grid">')

    for img in badcases["images"]:
        badge_class = f"badge-{img['badge']}"
        badge_text = {"fp": "FP Top", "fn": "FN Top", "both": "FP+FN Top"}[img["badge"]]

        fp_tags = " ".join(
            f'<span class="tag" style="background:#ffcdd2;">FP: {DISPLAY_NAMES[c]}</span>'
            for c in img.get("fp_classes", []))
        fn_tags = " ".join(
            f'<span class="tag" style="background:#ffe0b2;">FN: {DISPLAY_NAMES[c]}</span>'
            for c in img.get("fn_classes", []))

        parts.append(f"""
        <div class="badcase-card">
            <div class="info">
                <span class="badge {badge_class}">{badge_text}</span>
                FP={img['fp_total']} FN={img['fn_total']}
            </div>
            <div class="img-row">
                <img src="data:image/png;base64,{img['rgb_b64']}" alt="RGB" loading="lazy">
                <img src="data:image/png;base64,{img['gt_b64']}" alt="GT" loading="lazy">
                <img src="data:image/png;base64,{img['pred_b64']}" alt="Pred" loading="lazy">
            </div>
            <div class="info" style="font-size:10px;color:#666;word-break:break-all;">{img['image_rel']}</div>
            <div class="tags">{fp_tags} {fn_tags}</div>
        </div>""")

    parts.append("</div>")
    return "\n".join(parts)
    """Build the full analysis section HTML (overview, Precision/Recall findings, suggestions)."""

    s = summary["per_class"]
    g = summary["global"]

    # Data for the analysis
    # Find the "worst" attributes
    worst_prec = min(SEG_CLASS_NAMES, key=lambda n: s[n]["precision"])
    worst_rec  = min(SEG_CLASS_NAMES, key=lambda n: s[n]["recall"])

    # Build the diagnostic table
    diag_rows = ""
    for n in SEG_CLASS_NAMES:
        prec = s[n]["precision"]
        rec  = s[n]["recall"]
        f1   = s[n]["f1"]
        # Diagnosis
        issues = []
        if prec < 0.85:
            issues.append("FP较多")
        if rec < 0.80:
            issues.append("FN较多(漏检)")
        if prec >= 0.90 and rec >= 0.90:
            issues.append("良好")
        elif prec >= 0.85 and rec >= 0.80:
            issues.append("一般")
        if prec == 1.0 and rec == 1.0:
            issues.append("无问题")
        diag = "，".join(issues)
        diag_rows += f"""
            <tr>
                <td class="class-label">{DISPLAY_NAMES[n]}</td>
                <td>{prec:.4f}</td><td>{rec:.4f}</td><td>{f1:.4f}</td>
                <td>{diag}</td>
            </tr>"""

    return f"""
<div class="analysis-box">

<h3>4.1 整体表现概览</h3>
<table class="mini-table">
    <thead><tr><th>属性</th><th>Precision</th><th>Recall</th><th>F1</th><th>诊断</th></tr></thead>
    <tbody>{diag_rows}</tbody>
</table>
<p style="margin-top:10px;">
    {g['num_images']} 张验证图中，<b>77% 没有 FP</b>，<b>69% 没有 FN</b>。
    问题集中在少数属性和少数图片上。
</p>

<h3>4.2 Precision 分析：像素占比与检出精度呈显著正相关</h3>

<h4>① Upper / Lower — 占比 5% 是临界点</h4>
<p>
    当 <b>GT 像素占比 &lt; 5%</b> 时，Precision 从 1.0 骤降至 <b>0.72~0.76</b>，Recall 也降至 0.67~0.77。
    低占比意味着人物占比小（远景/小目标），模型容易把背景误判为衣物，或漏掉小区域衣物。
    当占比 &gt; 5% 后，Precision 稳定在 1.0，Recall 稳定在 0.97+。
</p>
<div class="highlight">
    <b>结论</b>：人体占比越大，上衣/下装的检出越可靠。<b>5% 像素占比</b>似乎是可靠检出的临界阈值。
</div>

<h4>② Glasses / Mask / Hat — 占比翻倍，Recall 大幅提升</h4>
<p>这些小属性像素占比极低（glasses 平均仅占图像的 0.1%），模型很难稳定检测到它们：</p>
<table class="mini-table">
    <thead><tr><th>属性</th><th>低占比组 Recall</th><th>高占比组 Recall</th><th>提升幅度</th></tr></thead>
    <tbody>
        <tr><td>glasses</td><td>0.500（≤0.38%）</td><td>0.718（>0.38%）</td><td>+44%</td></tr>
        <tr><td>mask</td><td>0.436（≤0.76%）</td><td>0.711（>0.76%）</td><td>+63%</td></tr>
        <tr><td>hat</td><td>0.637（≤1.6%）</td><td>0.906（>1.6%）</td><td>+42%</td></tr>
        <tr><td>dress</td><td>0.755（≤21%）</td><td>0.820（>21%）</td><td>+9%</td></tr>
    </tbody>
</table>
<div class="highlight">
    <b>结论</b>：小属性（glasses/mask/hat）的 Recall 随占比增加而显著提高，增幅达 40%~60%。
    这是模型对小目标分割能力的系统性局限。
</div>

<h4>③ Dress — FP 问题的根因：Upper + Lower → 误判为 Dress</h4>
<p>
    Dress 的 Precision 仅 <b>0.6605</b>，是所有属性中最差的。536 张图有 Dress FP：
</p>
<ul>
    <li><b>89%</b> 的 Dress FP 图中，GT 有 Upper</li>
    <li><b>88%</b> 的 Dress FP 图中，GT 有 Lower</li>
    <li><b>83%</b> 同时有 Upper 和 Lower</li>
    <li>仅 <b>7%</b>（36张）是完全没有上下装的</li>
</ul>
<div class="warn">
    <b>根因</b>：模型不是随机猜 Dress，而是当看到一个人同时穿着上衣和下装时，
    频繁将其合并误判为一整条连衣裙。尤其上下装颜色相近、边界不清晰时，模型倾向于预测 Dress。
    这是数据层面的混淆——上下分体装 vs 连衣裙本就是视觉上的 Hard Case。
</div>

<h4>④ Hair FP — 与 Hat 强相关</h4>
<p>
    Hair 有 517 个 FP。其中 <b>383 张（74%）</b> 同时有 Hat 在 GT 中出现。
    模型可能在部分 Hat 区域误判为 Hair。
</p>

<h3>4.3 Recall 分析：低占比属性的漏检规律</h3>
<p>
    Recall 分桶曲线揭示了<b>模型在什么条件下容易漏检（FN）</b>。
    与 Precision 不同，Recall 对像素占比的敏感性更强，尤其在小属性上。
</p>

<h4>① Glasses / Mask — Recall 随占比急速上升</h4>
<p>
    这两个最小属性的 Recall 曲线呈<b>陡峭爬升</b>形态：占比每增加一点，Recall 就显著提升。
    Glasses 的 GT 占比从 0.002 增加到 0.004 时，Recall 约从 0.40 上升到 0.55+。
    Mask 的趋势类似但样本更少，曲线波动更大。
    这验证了之前的结论：<b>小属性检出失败的主因是目标太小，模型根本看不到</b>。
</p>
<div class="highlight">
    <b>关键洞察</b>：Glasses 的全局 Recall 仅 0.61，不是因为模型不会识别眼镜，
    而是因为眼镜在大多数图片中占比太小（中位数仅 0.38%）。
    当眼镜占比 &gt; 0.5% 时，检出率显著提升至 70%+。
</div>

<h4>② Hat — Recall 分两段</h4>
<p>
    Hat 在 GT 占比 &lt; 1.6% 时 Recall 约 0.64，占比 &gt; 1.6% 后跃升至 0.91。
    <b>1.6% 像素占比</b>似乎是 Hat 可靠检出的关键拐点。
</p>

<h4>③ Dress — 占比很大时 Recall 依然不满</h4>
<p>
    Dress 即使在高占比情况下（>30% 图像面积），Recall 仍然只有 0.80 左右，无法达到接近 1.0。
    这与其他属性不同——Upper/Lower/Hat 在高占比时 Recall 都接近 1.0。
    可能原因：Dress 存在标签歧义（某些长上衣+半身裙的搭配，标注者可能意见不一致）。
</p>

<h4>④ Pred Ratio vs GT Ratio 的 Recall 差异</h4>
<p>
    对比 Recall 图中的绿线（GT 占比）和紫线（Pred 占比）：
</p>
<ul>
    <li><b>绿线（GT 占比 → Recall）</b>：反映「真实大小」的影响。曲线越陡，说明模型对目标大小的敏感度越高。</li>
    <li><b>紫线（Pred 占比 → Recall）</b>：反映「模型预测大小」与 Recall 的关系。如果紫线始终在绿线下方，说明模型倾向于<b>低估</b>该属性的面积。</li>
</ul>
<p>
    对于 Glasses，紫线整体低于绿线约 0.1-0.15，说明模型预测的眼镜区域系统性偏小。
</p>

<h3>4.4 分桶曲线总体解读方法</h3>
<ul>
    <li><b>Precision 图（蓝色/橙色）</b>：看模型会不会<b>误检</b>（FP）。曲线在高位且平稳 → 不会乱报。Dress 的 Precision 曲线在低占比区从 0 跃升到 1.0，说明大量 FP 集中在「GT 没有但 Pred 有」的区域。</li>
    <li><b>Recall 图（绿色/紫色）</b>：看模型会不会<b>漏检</b>（FN）。曲线从低占比到高占比单调上升 → 占比越大越不容易漏。曲线平直且不高 → 该属性本身难识别（如 Glasses）。</li>
    <li>标注在点旁的数字是<b>该桶的图片数量</b>，可判断统计显著性——样本少的桶（&lt;50），指标不具参考价值。</li>
</ul>

<h3>4.5 改善建议</h3>
<table class="mini-table">
    <thead><tr><th>问题</th><th>建议</th></tr></thead>
    <tbody>
        <tr>
            <td>小属性（glasses/mask）Recall 低</td>
            <td>增加小类样本量（mask 仅 298 张 GT present）；对小类使用 Focal Loss 或更大 loss weight</td>
        </tr>
        <tr>
            <td>Dress FP 高</td>
            <td>增加难例（上下装颜色相近的样本）做针对性训练；或加入显式的「Dress vs Upper+Lower」二分类监督</td>
        </tr>
        <tr>
            <td>低占比时 Upper/Lower Precision 骤降</td>
            <td>增加多尺度训练（当前仅 256×256）；使用小目标增强（随机缩小后拼接）</td>
        </tr>
        <tr>
            <td>整体</td>
            <td>检出问题本质上就是分割质量问题——分割好了自然能检出。提升 mIoU 是根本方向</td>
        </tr>
    </tbody>
</table>

</div>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--badcase_dir", type=str, default=None)
    args = p.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv_path) if args.csv_path else base_dir / "per_image_detection.csv"
    badcase_dir = Path(args.badcase_dir) if args.badcase_dir else base_dir / "badcase"
    output_path = Path(args.output) if args.output else base_dir / "detection_analysis_report.html"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run predict_detection.py first.")
        sys.exit(1)

    print("Loading CSV...")
    rows = load_csv(csv_path)
    print(f"  {len(rows)} images loaded")

    print("Computing summary...")
    summary = compute_summary(rows)
    print(f"  Avg mIoU: {summary['global']['mIoU_avg']:.4f}")
    for n in SEG_CLASS_NAMES:
        s = summary["per_class"][n]
        print(f"  {n:12s}: P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f}")

    print("Computing Precision bucket stats...")
    buckets = compute_all_buckets(rows)

    print("Computing Recall bucket stats...")
    recall_buckets = compute_all_recall_buckets(rows)

    print("Computing Absolute Pixel bucket stats...")
    abs_buckets = compute_all_abs_pixel_buckets(rows)

    print("Generating matplotlib Ratio Precision charts...")
    chart_images = {}
    for n in SEG_CLASS_NAMES:
        chart_images[n] = make_bucket_chart(n, buckets)
    print(f"  {len(chart_images)} Ratio Precision charts rendered")

    print("Generating matplotlib Absolute Pixel Precision charts...")
    abs_chart_images = {}
    for n in SEG_CLASS_NAMES:
        abs_chart_images[n] = make_abs_pixel_chart(n, abs_buckets)
    print(f"  {len(abs_chart_images)} Absolute Pixel Precision charts rendered")

    print("Generating matplotlib Ratio Recall charts...")
    recall_chart_images = {}
    for n in SEG_CLASS_NAMES:
        recall_chart_images[n] = make_recall_bucket_chart(n, recall_buckets)
    print(f"  {len(recall_chart_images)} Recall charts rendered")

    print("Generating Trimmed Ratio Precision charts (95th pct, 20 bins)...")
    trimmed_chart_images = {}
    for n in SEG_CLASS_NAMES:
        gt_trim = compute_trimmed_buckets(rows, n, "gt_ratio")
        pred_trim = compute_trimmed_buckets(rows, n, "pred_ratio")
        trimmed_chart_images[n] = make_trimmed_ratio_chart(n, gt_trim, pred_trim)
    print(f"  {len(trimmed_chart_images)} Trimmed ratio charts rendered")

    print("Generating Trimmed Absolute Pixel Precision charts (95th pct, 20 bins)...")
    trimmed_abs_chart_images = {}
    for n in SEG_CLASS_NAMES:
        gt_trim = compute_trimmed_abs_buckets(rows, n, "gt_pixels")
        pred_trim = compute_trimmed_abs_buckets(rows, n, "pred_pixels")
        trimmed_abs_chart_images[n] = make_trimmed_abs_chart(n, gt_trim, pred_trim)
    print(f"  {len(trimmed_abs_chart_images)} Trimmed abs charts rendered")

    print("Loading badcases...")
    badcases = load_badcases(badcase_dir)
    if badcases:
        print(f"  {len(badcases['images'])} badcase images loaded")
    else:
        print("  No badcase data found")

    print("Generating HTML...")
    generate_html(summary, chart_images, recall_chart_images, abs_chart_images,
                  trimmed_chart_images, trimmed_abs_chart_images, badcases, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
