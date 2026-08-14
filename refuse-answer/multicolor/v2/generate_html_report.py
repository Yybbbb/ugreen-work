#!/usr/bin/env python3
import argparse
import base64
import html
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from image_io import probe_image_size
from mask_io import load_upper_mask_for_image


DEFAULT_INPUT = Path(__file__).resolve().parent / "eval_val_pure_multicolor.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "eval_report.html"


def generate_report(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, max_badcases=120, seed=42):
    input_path = Path(input_path)
    output_path = Path(output_path)
    with open(input_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    output_path.write_text(_render_html(report, max_badcases=max_badcases, seed=seed), encoding="utf-8")
    return output_path


def _render_html(report, max_badcases=120, seed=42):
    cm = report["confusion_matrix"]
    metrics = report["metrics_processed_only"]
    sets = report["sets"]
    results = report["results"]
    experiment = report.get("experiment", "v2")
    title = f"{experiment} 杂色门控评估报告"

    total = sum(item["total"] for item in sets.values())
    processed = sum(item["processed"] for item in sets.values())
    errors = sum(item["errors"] for item in sets.values())
    correct = cm["actual_pure_pred_pure_tp"] + cm["actual_multi_pred_multi_tn"]
    accuracy_with_errors = correct / total if total else 0.0

    fn_pure = [
        item for item in results
        if item.get("gt_label") == "pure" and item.get("prediction") == "multi"
    ]
    fp_multi = [
        item for item in results
        if item.get("gt_label") == "multi" and item.get("prediction") == "pure"
    ]
    errors_items = [item for item in results if "error" in item]

    fn_total = len(fn_pure)
    fp_total = len(fp_multi)
    fn_pure = _sample_half(fn_pure, seed + 1)[:max_badcases]
    fp_multi = _sample_half(fp_multi, seed + 2)[:max_badcases]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #20242a; }}
    header {{ padding: 24px 32px; background: #1f2937; color: white; }}
    main {{ padding: 24px 32px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; }}
    h3 {{ margin: 20px 0 10px; font-size: 16px; }}
    .subtle {{ color: #6b7280; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ background: white; border: 1px solid #d8dde6; border-radius: 8px; padding: 14px; }}
    .metric .label {{ color: #5b6472; font-size: 13px; }}
    .metric .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; border: 1px solid #d8dde6; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef1f5; }}
    .badcase-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #d8dde6; border-radius: 8px; overflow: hidden; }}
    .image-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #d8dde6; }}
    .image-panel {{ background: #111827; }}
    .image-panel .caption {{ color: #d1d5db; font-size: 12px; padding: 5px 8px; }}
    .card img {{ width: 100%; height: 240px; object-fit: contain; background: #111827; display: block; }}
    .card-body {{ padding: 10px 12px; font-size: 13px; line-height: 1.45; }}
    .tag {{ display: inline-block; padding: 2px 6px; border-radius: 999px; background: #e5e7eb; margin-right: 4px; }}
    .tag.bad {{ background: #fee2e2; color: #991b1b; }}
    .tag.warn {{ background: #fef3c7; color: #92400e; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; word-break: break-all; }}
    details {{ margin-top: 10px; }}
    summary {{ cursor: pointer; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div>真值来源：<span class="mono">val_pure</span> 视为纯色，<span class="mono">val_multicolor/upper_multicolor</span> 视为杂色；仅检测 upper mask。</div>
  </header>
  <main>
    <h2>总体指标</h2>
    <div class="grid">
      {_metric("Processed / Total", f"{processed} / {total}")}
      {_metric("Errors", str(errors))}
      {_metric("Accuracy", _pct(metrics["accuracy"]))}
      {_metric("Accuracy Errors Wrong", _pct(accuracy_with_errors))}
      {_metric("Pure Recall", _pct(metrics["pure_recall"]))}
      {_metric("Multi Recall", _pct(metrics["multi_recall"]))}
      {_metric("Pure Precision", _pct(metrics["pure_precision"]))}
      {_metric("Multi Precision", _pct(metrics["multi_precision"]))}
    </div>

    <h2>集合覆盖</h2>
    <table>
      <tr><th>集合</th><th>Total</th><th>Processed</th><th>Errors</th><th>Coverage</th></tr>
      {_set_row("val_pure", sets["pure"])}
      {_set_row("val_multicolor/upper_multicolor", sets["multi"])}
    </table>

    <h2>Confusion Matrix</h2>
    <table>
      <tr><th>Actual \\ Predicted</th><th>Pred Pure</th><th>Pred Multi</th></tr>
      <tr><th>Actual Pure</th><td>{cm["actual_pure_pred_pure_tp"]}</td><td>{cm["actual_pure_pred_multi_fn"]}</td></tr>
      <tr><th>Actual Multi</th><td>{cm["actual_multi_pred_pure_fp"]}</td><td>{cm["actual_multi_pred_multi_tn"]}</td></tr>
    </table>

    <h2>Badcases</h2>
    <h3>Actual Multi / Pred Pure ({len(fp_multi)} randomly sampled, total {fp_total})</h3>
    <div class="badcase-grid">{''.join(_badcase_card(item, "Actual Multi / Pred Pure") for item in fp_multi)}</div>

    <h3>Actual Pure / Pred Multi ({len(fn_pure)} randomly sampled, total {fn_total})</h3>
    <div class="badcase-grid">{''.join(_badcase_card(item, "Actual Pure / Pred Multi") for item in fn_pure)}</div>

    <h2>Errors ({len(errors_items)})</h2>
    <table>
      <tr><th>ID</th><th>GT</th><th>Error</th><th>Path</th></tr>
      {''.join(_error_row(item) for item in errors_items)}
    </table>
  </main>
</body>
</html>
"""


def _metric(label, value):
    return f'<div class="metric"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'


def _set_row(name, item):
    coverage = item["processed"] / item["total"] if item["total"] else 0.0
    return (
        f"<tr><td>{html.escape(name)}</td><td>{item['total']}</td>"
        f"<td>{item['processed']}</td><td>{item['errors']}</td><td>{_pct(coverage)}</td></tr>"
    )


def _badcase_card(item, title):
    d = item["detection"]
    path = item.get("path", "")
    image_src = _image_data_uri(path)
    mask_src = _upper_mask_data_uri(item)
    return f"""
<div class="card">
  <div class="image-pair">
    <div class="image-panel"><div class="caption">Original</div><img src="{html.escape(image_src)}" alt="{html.escape(item.get('id', 'image'))}"></div>
    <div class="image-panel"><div class="caption">Upper Mask</div><img src="{html.escape(mask_src)}" alt="upper mask"></div>
  </div>
  <div class="card-body">
    <div><span class="tag bad">{html.escape(title)}</span><span class="tag warn">{html.escape(d.get('decision_reason', ''))}</span></div>
    <div class="mono">{html.escape(item.get('id', ''))}</div>
    <div>main={d.get('main_ratio')} second={d.get('second_ratio')} minor={d.get('minor_total_ratio')} n_effective={d.get('n_effective_colors')}</div>
    <div>n_pixels={d.get('n_pixels')}</div>
    <details><summary>paths</summary><div class="mono">{html.escape(path)}</div><div class="mono">{html.escape(item.get('annotation_path', ''))}</div></details>
  </div>
</div>
"""


def _error_row(item):
    return (
        f"<tr><td>{html.escape(item.get('id', ''))}</td>"
        f"<td>{html.escape(item.get('gt_label', ''))}</td>"
        f"<td>{html.escape(item.get('error', ''))}</td>"
        f"<td class=\"mono\">{html.escape(item.get('path', ''))}</td></tr>"
    )


def _sample_half(items, seed):
    if not items:
        return []
    sample_size = max(1, math.ceil(len(items) / 2.0))
    rng = random.Random(seed)
    sampled = rng.sample(list(items), sample_size)
    return sorted(sampled, key=lambda item: item.get("id", ""))


def _image_data_uri(path):
    path_obj = Path(path)
    mime = _mime_type(path_obj)
    try:
        encoded = base64.b64encode(path_obj.read_bytes()).decode("ascii")
    except OSError:
        encoded = ""
        mime = "text/plain"
    return f"data:{mime};base64,{encoded}"


def _upper_mask_data_uri(item):
    path = item.get("path", "")
    try:
        width, height = probe_image_size(path)
        mask_info = load_upper_mask_for_image(path)
        mask = mask_info["mask"]
        if mask_info["width"] != width or mask_info["height"] != height:
            mask = _resize_mask_nearest(mask, width, height)
        svg = _mask_svg(mask, width, height)
    except Exception as exc:
        svg = _placeholder_mask_svg(str(exc))
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _mask_svg(mask, width, height):
    max_dim = 320
    scale = max(1, int(max(width, height) / max_dim))
    rects = []
    for y in range(0, height, scale):
        row = mask[y]
        x = 0
        while x < width:
            if row[x]:
                start = x
                while x < width and row[x]:
                    x += scale
                rect_w = max(scale, x - start)
                rects.append(f'<rect x="{start}" y="{y}" width="{rect_w}" height="{scale}"/>')
            else:
                x += scale
    body = "".join(rects)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="#111827"/>'
        f'<g fill="#22c55e" opacity="0.88">{body}</g>'
        "</svg>"
    )


def _placeholder_mask_svg(message):
    safe = html.escape(message[:120])
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 240">'
        '<rect width="320" height="240" fill="#111827"/>'
        '<text x="16" y="118" fill="#fca5a5" font-size="14">mask unavailable</text>'
        f'<text x="16" y="140" fill="#9ca3af" font-size="10">{safe}</text>'
        "</svg>"
    )


def _resize_mask_nearest(mask, target_width, target_height):
    src_height = len(mask)
    src_width = len(mask[0]) if src_height else 0
    if src_width == 0 or src_height == 0:
        return [[0 for _ in range(target_width)] for _ in range(target_height)]

    resized = []
    for y in range(target_height):
        src_y = min(src_height - 1, int(y * src_height / target_height))
        row = []
        for x in range(target_width):
            src_x = min(src_width - 1, int(x * src_width / target_width))
            row.append(mask[src_y][src_x])
        resized.append(row)
    return resized


def _mime_type(path):
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _pct(value):
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Generate V2 evaluation HTML report.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-badcases", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_path = generate_report(args.input, args.output, max_badcases=args.max_badcases, seed=args.seed)
    print(output_path)


if __name__ == "__main__":
    main()
