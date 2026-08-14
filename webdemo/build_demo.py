#!/usr/bin/env python3
"""Build the V4A test-set attribute demo HTML.

Reads the V4A (replay1k) final test inference + Qwen attribute evaluation,
computes a per-sample strict Micro F1 (same scoring as
``evaluate_qwen_attribute_extraction.score_attribute_pair``), bins samples into
5 F1 tiers, randomly samples up to 20 per tier, and emits a self-contained
``index.html`` plus ``samples.json`` (image paths for the server) and
``metrics.json``.

Run:
    python webdemo/build_demo.py
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Reuse the project's official scoring so per-sample F1 matches metrics.json.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_qwen_attribute_extraction import (  # noqa: E402
    SCALAR_FIELDS,
    _nested,
    _prf,
    deduplicate_rows,
    iter_jsonl,
    normalize_value,
    score_attribute_pair,
)

DEFAULT_RUN_DIR = PROJECT_ROOT / "artifacts/sft/region_category_person_sft_30k_replay1k_b16_e3_lr125/test_inference_final"
DEFAULT_QWEN_GT = PROJECT_ROOT / "data" / "prepared" / "test_qwen_attributes.jsonl"
SEED = 20260807
PER_TIER = 20

# F1 tier boundaries (inclusive lower, exclusive upper except top).
# T1 worst .. T5 best.
TIERS = [
    {"id": 5, "label": "挡位5 · 优秀", "lo": 0.8, "hi": 1.01, "stars": "★★★★★"},
    {"id": 4, "label": "挡位4 · 良好", "lo": 0.6, "hi": 0.8, "stars": "★★★★"},
    {"id": 3, "label": "挡位3 · 中等", "lo": 0.4, "hi": 0.6, "stars": "★★★"},
    {"id": 2, "label": "挡位2 · 较差", "lo": 0.2, "hi": 0.4, "stars": "★★"},
    {"id": 1, "label": "挡位1 · 很差", "lo": 0.0, "hi": 0.2, "stars": "★"},
]

FIELD_LABELS = {
    "age_group": "年龄",
    "gender": "性别",
    "upper_garment.type": "上衣类型",
    "upper_garment.color": "上衣颜色",
    "upper_garment.length": "上衣袖长",
    "lower_garment.type": "下衣类型",
    "lower_garment.color": "下衣颜色",
    "lower_garment.length": "下衣长度",
    "shoes.type": "鞋类型",
    "shoes.color": "鞋颜色",
    "head.accessories": "头部配饰",
    "head.hairstyle": "发型",
    "head.hair_color": "发色",
    "head.hair_length": "发长",
    "carried_items.handbag": "手提包",
    "carried_items.backpack": "背包",
    "handheld_items.dangerous_item": "危险物",
    "handheld_items.mobile_phone": "手机",
}


def tier_of(f1: float) -> Optional[Dict[str, Any]]:
    for t in TIERS:
        if t["lo"] <= f1 < t["hi"]:
            return t
    # f1 == 0 lands in T1
    return TIERS[-1]


def field_status(gt: Mapping[str, Any], pred: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for field in SCALAR_FIELDS:
        g_raw = _nested(gt, field)
        p_raw = _nested(pred, field)
        g = normalize_value(field, g_raw)
        p = normalize_value(field, p_raw)
        if g is None and p is None:
            status = "both_unknown"
        elif g is None and p is not None:
            status = "extra"  # GT unknown, model emitted -> unknown-extra
        elif g is not None and p is None:
            status = "miss"  # GT known, model missed
        elif g == p:
            status = "match"
        else:
            status = "mismatch"
        rows.append({
            "field": field,
            "label": FIELD_LABELS.get(field, field),
            "gt": g_raw if g_raw is not None else "",
            "pred": p_raw if p_raw is not None else "",
            "gt_n": g if g is not None else "",
            "pred_n": p if p is not None else "",
            "status": status,
        })
    return rows


def build_samples(run_dir: Path, qwen_gt_path: Path = DEFAULT_QWEN_GT) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Use the Qwen-GT basis: GT attributes are Qwen's extraction from the GT caption
    # (test_qwen_attributes.jsonl), so pred and GT both pass through the same extractor.
    metrics = json.loads((run_dir / "evaluation_qwen_final" / "metrics_qwen_gt.json").read_text(encoding="utf-8"))
    preds = {r["sample_id"]: r for r in iter_jsonl(run_dir / "predictions.jsonl")}
    dedup = deduplicate_rows(list(iter_jsonl(run_dir / "evaluation_qwen_final" / "qwen_attributes.dedup.jsonl")))
    qwen_gt = {}
    for r in iter_jsonl(qwen_gt_path):
        sid = str(r.get("sample_id") or r.get("id") or "")
        attrs = r.get("qwen_attributes")
        if sid and isinstance(attrs, Mapping):
            qwen_gt[sid] = attrs

    all_items: List[Dict[str, Any]] = []
    for sid, row in dedup.items():
        p = preds.get(sid)
        if p is None:
            continue
        gt = qwen_gt.get(sid, {})
        if not isinstance(gt, Mapping):
            gt = {}
        pr = row.get("attributes") if isinstance(row.get("attributes"), Mapping) else {}
        counts = score_attribute_pair(gt, pr)
        f1 = _prf(counts["tp"], counts["fp"], counts["fn"])["f1"]
        all_items.append({
            "sample_id": sid,
            "f1": round(f1, 4),
            "image": p.get("image", ""),
            "bbox_xyxy": p.get("bbox_xyxy"),
            "scene": p.get("scene", ""),
            "session": p.get("session", ""),
            "scale": p.get("scale", ""),
            "gt_caption": p.get("label", ""),
            "pred_caption": row.get("prediction", p.get("prediction", "")),
            "gt_attrs": gt,
            "pred_attrs": pr,
            "fields": field_status(gt, pr),
        })

    # Bin into tiers.
    buckets: Dict[int, List[Dict[str, Any]]] = {t["id"]: [] for t in TIERS}
    for it in all_items:
        t = tier_of(it["f1"])
        if t is None:
            continue
        buckets[t["id"]].append(it)

    rng = random.Random(SEED)
    sampled: List[Dict[str, Any]] = []
    tier_summary = []
    # Display best tier first.
    for t in TIERS:
        pool = buckets[t["id"]]
        pool_sorted = sorted(pool, key=lambda x: x["f1"], reverse=(t["id"] >= 3))
        picks = pool_sorted if len(pool_sorted) <= PER_TIER else rng.sample(pool_sorted, PER_TIER)
        for it in picks:
            it["tier"] = t["id"]
            it["tier_label"] = t["label"]
            it["tier_stars"] = t["stars"]
            sampled.append(it)
        tier_summary.append({
            "id": t["id"],
            "label": t["label"],
            "stars": t["stars"],
            "lo": t["lo"],
            "hi": t["hi"],
            "total": len(pool),
            "sampled": len(picks),
        })

    # Assign a stable global index for /img/{i} serving.
    for i, it in enumerate(sampled):
        it["i"] = i

    return sampled, {"metrics": metrics, "tiers": tier_summary, "total_samples": len(all_items)}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Florence Person Attribute · V4A 测试集 Demo</title>
<style>
:root{
  --bg:#0f1115; --panel:#171a21; --panel2:#1f2430; --ink:#e6e9ef; --mut:#8b93a4;
  --line:#2a3140; --accent:#6ea8fe; --ok:#3dd68c; --bad:#ff6b6b; --warn:#ffb454; --unk:#5b6273;
  --t5:#3dd68c; --t4:#9ece6a; --t3:#e0af68; --t2:#ff9e64; --t1:#f7768e;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.5}
a{color:var(--accent)}
.wrap{max-width:1480px;margin:0 auto;padding:24px 20px 80px}
header.hero{padding:18px 0 10px;border-bottom:1px solid var(--line);margin-bottom:18px}
header.hero h1{margin:0 0 4px;font-size:22px;font-weight:650}
header.hero .sub{color:var(--mut);font-size:13px}
h2.section{font-size:16px;margin:28px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px}
h2.section .pill{font-size:11px;color:var(--mut);font-weight:500;background:var(--panel2);padding:2px 8px;border-radius:10px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.metric .k{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.metric .v{font-size:22px;font-weight:680;margin-top:3px}
.metric .v small{font-size:12px;color:var(--mut);font-weight:400}
.metric .sub{font-size:11px;color:var(--mut);margin-top:2px}
.fieldbars{display:grid;grid-template-columns:repeat(2,1fr);gap:4px 22px;margin-top:8px}
.fbar{display:grid;grid-template-columns:96px 1fr 42px;align-items:center;gap:8px;font-size:11px}
.fbar .name{color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fbar .track{background:var(--panel2);height:7px;border-radius:4px;overflow:hidden}
.fbar .fill{height:100%;border-radius:4px}
.fbar .num{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
.tier-nav{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 4px;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5}
.tier-nav a{text-decoration:none;font-size:12px;padding:5px 11px;border-radius:14px;border:1px solid var(--line);color:var(--ink);background:var(--panel)}
.tier-nav a:hover{border-color:var(--accent)}
.tier-head{display:flex;align-items:center;gap:12px;margin:30px 0 12px}
.tier-head .badge{font-size:13px;font-weight:650;padding:4px 12px;border-radius:8px;color:#0f1115}
.tier-head .range{font-size:12px;color:var(--mut)}
.tier-head .count{font-size:12px;color:var(--mut);margin-left:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.card .imgwrap{position:relative;background:#000;aspect-ratio:16/9;overflow:hidden}
.card .imgwrap img{width:100%;height:100%;object-fit:contain;display:block}
.card .f1tag{position:absolute;top:8px;left:8px;font-size:12px;font-weight:700;padding:3px 9px;border-radius:6px;color:#0f1115}
.card .sid{position:absolute;bottom:6px;left:8px;font-size:10px;color:#cfd3dc;background:rgba(0,0,0,.55);padding:2px 6px;border-radius:4px;max-width:90%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .body{padding:11px 13px 13px;display:flex;flex-direction:column;gap:9px}
.capblock{font-size:13.5px}
.capblock .lab{font-size:10.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px}
.capblock.pred .txt{color:#cfe1ff;border-left:3px solid var(--accent);padding-left:8px}
.capblock.gt .txt{color:var(--ink);border-left:3px solid var(--mut);padding-left:8px}
.attrs{font-size:10.5px;border-collapse:collapse;width:100%}
.attrs th,.attrs td{padding:2px 5px;border-bottom:1px solid var(--line);vertical-align:top}
.attrs th{text-align:left;color:var(--mut);font-weight:500;font-size:10px}
.attrs td.f{color:var(--mut);white-space:nowrap}
.attrs td.gt{color:var(--ink)}
.attrs td.pred{color:var(--ink)}
.attrs tr.match td.pred{color:var(--ok)}
.attrs tr.mismatch td.pred{color:var(--bad)}
.attrs tr.miss td.pred{color:var(--warn)}
.attrs tr.extra td.pred{color:var(--warn)}
.attrs tr.both_unknown td{color:var(--unk)}
.attrs .st{font-size:9px}
footer{color:var(--mut);font-size:11px;margin-top:34px;border-top:1px solid var(--line);padding-top:12px}
.tcolor-5{background:var(--t5)} .tcolor-4{background:var(--t4)} .tcolor-3{background:var(--t3)} .tcolor-2{background:var(--t2)} .tcolor-1{background:var(--t1)}
</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <h1>Florence Person Attribute · V4A 测试集结果 Demo <span style="font-size:13px;color:var(--accent)">（Qwen-GT 基准）</span></h1>
  <div class="sub">checkpoint: <code>region_category_person_sft_30k_replay1k_b16_e3_lr125/final</code> · 4,328 条测试样本 · GT 与预测属性均由 <code>Qwen36-35b-caption</code> 抽取（GT 取自 GT caption，预测取自 Florence caption），同口径对比 · per-sample 严格 Micro F1 分 5 挡，每挡随机抽取 20 张（不足则全取）</div>
</header>

<h2 class="section">V4A 测试集核心指标 <span class="pill" id="m-pill"></span></h2>
<div style="font-size:12px;color:var(--mut);margin:-6px 0 10px">GT = Qwen 从 GT caption 抽取的属性（<code>test_qwen_attributes.jsonl</code>）；预测 = Qwen 从 Florence 推理 caption 抽取的属性。指标来自 <code>metrics_qwen_gt.json</code>。</div>
<div class="metrics" id="m-main"></div>

<h2 class="section">逐字段 F1 <span class="pill">18 标量字段</span></h2>
<div class="fieldbars" id="m-fields"></div>

<div class="tier-nav" id="nav"></div>
<div id="tiers"></div>

<footer>
  每张图展示：原图（红框为人物检测框）· Florence 推理 caption · GT caption · 属性逐字段对比（绿=匹配，红=不匹配，橙=漏标/多余，灰=双方未知）。
  GT 属性与 Pred 属性均由 <code>Qwen36-35b-caption</code> 从各自 caption 抽取（同口径），per-sample F1 复用 <code>evaluate_qwen_attribute_extraction.score_attribute_pair</code> 的严格 Micro F1（tp/fp/fn over 已知属性+extra）。
</footer>
</div>
<script>
const DATA = __SAMPLES_JSON__;
const META = __META_JSON__;

function pct(x){return (x*100).toFixed(2)+'%';}
function f4(x){return (x).toFixed(4);}

(function renderMetrics(){
  const m=META.metrics, cap=m.caption;
  const main=[
    ['Micro F1', f4(m.micro.f1), `${f4(m.micro.precision)} P / ${f4(m.micro.recall)} R`],
    ['Soft Micro F1', f4(m.soft_micro.f1), `${f4(m.soft_micro.precision)} P / ${f4(m.soft_micro.recall)} R`],
    ['Macro-field F1', f4(m.macro_field_f1), '18 字段等权平均'],
    ['Mean field exact', f4(m.mean_field_exact), '已知字段精确匹配率'],
    ['Unknown-extra', pct(m.unknown_extra_rate), `${m.unknown_extra_count} / ${m.unknown_slots} 槽位`],
    ['平均词数', cap.average_words.toFixed(2), `P95=${cap.p95_words} · 18-24词 ${pct(cap.length_18_24_ratio)}`],
    ['背景关键词率', pct(cap.background_keyword_ratio), `单句率 ${pct(cap.single_sentence_ratio)}`],
    ['抽取失败', m.extractor_failures, `输入行 ${m.input_lines} · 样本 ${m.samples}`],
  ];
  document.getElementById('m-pill').textContent='测试样本 '+m.samples;
  document.getElementById('m-main').innerHTML=main.map(r=>
    `<div class="metric"><div class="k">${r[0]}</div><div class="v">${r[1]}</div><div class="sub">${r[2]}</div></div>`).join('');

  const pf=m.per_field;
  const fields=__FIELDS_ORDER__;
  const max=Math.max(...fields.map(f=>pf[f].f1));
  document.getElementById('m-fields').innerHTML=fields.map(f=>{
    const v=pf[f].f1; const w=max>0?(v/max*100):0;
    const col = v>=0.7?'var(--ok)':v>=0.45?'var(--warn)':'var(--bad)';
    return `<div class="fbar"><div class="name" title="${f}">${__FIELD_LABELS__[f]||f}</div>`+
      `<div class="track"><div class="fill" style="width:${w}%;background:${col}"></div></div>`+
      `<div class="num">${v.toFixed(3)}</div></div>`;
  }).join('');
})();

(function renderNav(){
  document.getElementById('nav').innerHTML=META.tiers.map(t=>
    `<a href="#tier-${t.id}">${t.stars} 挡位${t.id} <span style="color:var(--mut)">(${t.sampled}/${t.total})</span></a>`).join('');
})();

(function renderTiers(){
  const byTier={};
  DATA.forEach(s=>{(byTier[s.tier]=byTier[s.tier]||[]).push(s);});
  const order=META.tiers.map(t=>t.id);
  let html='';
  order.forEach(id=>{
    const t=META.tiers.find(x=>x.id===id);
    const items=byTier[id]||[];
    html+=`<div id="tier-${id}">`;
    html+=`<div class="tier-head"><span class="badge tcolor-${id}">${t.stars} 挡位${id}</span>`+
      `<span class="range">F1 ∈ ${t.lo.toFixed(1)}–${t.hi===1.01?'1.0':t.hi.toFixed(1)} · ${t.label.split('·')[1].trim()}</span>`+
      `<span class="count">该挡共 ${t.total} 条 · 展示 ${items.length}</span></div>`;
    html+=`<div class="grid">`;
    items.forEach(s=>{
      const extraGt=(s.gt_attrs.extra||[]); const extraPr=(s.pred_attrs.extra||[]);
      const rows=s.fields.map(f=>{
        const st=f.status;
        const sym = st==='match'?'✓':st==='mismatch'?'✗':st==='miss'?'∅':st==='extra'?'+':'–';
        return `<tr class="${st}"><td class="f">${f.label}</td>`+
          `<td class="gt">${esc(f.gt_n||'—')}</td>`+
          `<td class="pred">${esc(f.pred_n||'—')} <span class="st">${sym}</span></td></tr>`;
      }).join('');
      html+=`<div class="card">
        <div class="imgwrap">
          <img loading="lazy" src="/img/${s.i}" alt="">
          <span class="f1tag tcolor-${id}">F1 ${s.f1.toFixed(3)}</span>
          <span class="sid">${esc(s.sample_id)}</span>
        </div>
        <div class="body">
          <div class="capblock pred"><div class="lab">Florence 推理 caption</div><div class="txt">${esc(s.pred_caption)}</div></div>
          <div class="capblock gt"><div class="lab">GT caption</div><div class="txt">${esc(s.gt_caption)}</div></div>
          <table class="attrs">
            <tr><th>字段</th><th>GT(qwen)</th><th>Pred(qwen)</th></tr>
            ${rows}
            <tr><td class="f">extra</td><td class="gt">${esc(extraGt.join('; ')||'—')}</td><td class="pred">${esc(extraPr.join('; ')||'—')}</td></tr>
          </table>
        </div>
      </div>`;
    });
    html+=`</div></div>`;
  });
  document.getElementById('tiers').innerHTML=html;
})();

function esc(s){s=(s===null||s===undefined)?'':String(s);return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
</script>
</body>
</html>
"""


def render_html(samples: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    fields_order = list(SCALAR_FIELDS)
    field_labels = {f: FIELD_LABELS.get(f, f) for f in fields_order}
    doc = (HTML_TEMPLATE
           .replace("__SAMPLES_JSON__", json.dumps(samples, ensure_ascii=False))
           .replace("__META_JSON__", json.dumps(meta, ensure_ascii=False))
           .replace("__FIELDS_ORDER__", json.dumps(fields_order))
           .replace("__FIELD_LABELS__", json.dumps(field_labels, ensure_ascii=False)))
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                        help="V4A test_inference_final directory")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "build",
                        help="output directory for index.html / samples.json / metrics.json")
    args = parser.parse_args()

    samples, meta = build_samples(args.run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # samples.json: only what the image server needs, in global-index order.
    server_samples = [{"image": s["image"], "bbox_xyxy": s["bbox_xyxy"]} for s in samples]
    (args.out_dir / "samples.json").write_text(json.dumps(server_samples, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "index.html").write_text(render_html(samples, meta), encoding="utf-8")

    print(f"built {len(samples)} sample cards across {len(meta['tiers'])} tiers")
    for t in meta["tiers"]:
        print(f"  挡位{t['id']} {t['stars']}: total={t['total']} sampled={t['sampled']}")
    print(f"output: {args.out_dir}")


if __name__ == "__main__":
    main()
