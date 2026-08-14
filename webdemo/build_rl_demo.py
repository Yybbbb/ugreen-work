#!/usr/bin/env python3
"""Build the RL checkpoint comparison demo.

Reads V4B SFT, qwen_rl, and lexical_rl evaluation artifacts, bins samples
into 5 tiers by V4B lexical soft-F1, samples up to 20 per tier, and emits
a self-contained index.html + samples.json for the image server.

Run:
    cd /data1/work/MichaelYu/florence-attibute
    python webdemo/build_rl_demo.py
    python webdemo/rl_server.py --port 8013
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "artifacts/rl/evaluation"
TEST_JSONL = PROJECT_ROOT / "data/prepared/test.jsonl"
SEED = 20260812
PER_TIER = 20
CANDIDATES = ["v4b_sft", "qwen_rl", "lexical_rl"]
CANDIDATE_LABELS = {
    "v4b_sft":    "V4B SFT（基线）",
    "qwen_rl":    "qwen_rl",
    "lexical_rl": "lexical_rl",
}

TIERS = [
    {"id": 5, "label": "挡位5 · 优秀", "lo": 0.8, "hi": 1.01, "stars": "★★★★★"},
    {"id": 4, "label": "挡位4 · 良好", "lo": 0.6, "hi": 0.8,  "stars": "★★★★"},
    {"id": 3, "label": "挡位3 · 中等", "lo": 0.4, "hi": 0.6,  "stars": "★★★"},
    {"id": 2, "label": "挡位2 · 较差", "lo": 0.2, "hi": 0.4,  "stars": "★★"},
    {"id": 1, "label": "挡位1 · 很差", "lo": 0.0, "hi": 0.2,  "stars": "★"},
]

FIELD_LABELS: Dict[str, str] = {
    "age_group":                  "年龄",
    "gender":                     "性别",
    "upper_garment.type":         "上衣类型",
    "upper_garment.color":        "上衣颜色",
    "upper_garment.length":       "上衣袖长",
    "lower_garment.type":         "下衣类型",
    "lower_garment.color":        "下衣颜色",
    "lower_garment.length":       "下衣长度",
    "shoes.type":                 "鞋类型",
    "shoes.color":                "鞋颜色",
    "head.accessories":           "头饰",
    "head.hairstyle":             "发型",
    "head.hair_color":            "发色",
    "head.hair_length":           "发长",
    "carried_items.handbag":      "手提包",
    "carried_items.backpack":     "背包",
    "handheld_items.dangerous_item": "危险物",
    "handheld_items.mobile_phone":   "手机",
}
FIELDS_ORDER = list(FIELD_LABELS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tier_of(f1: float) -> Dict[str, Any]:
    for t in TIERS:
        if t["lo"] <= f1 < t["hi"]:
            return t
    return TIERS[-1]


def _nested(d: Mapping, dotkey: str):
    parts = dotkey.split(".", 1)
    v = d.get(parts[0])
    if len(parts) == 1 or not isinstance(v, Mapping):
        return v
    return _nested(v, parts[1])


def _norm(s: Any) -> str:
    return str(s).strip().lower()


def flatten_gt(v: Any) -> str:
    """Render a GT value for display; list values joined with '/'."""
    if v is None:
        return ""
    if isinstance(v, list):
        parts = [str(x).strip() for x in v if x is not None and str(x).strip()]
        return "/".join(parts)
    return str(v)


def _list_hit(gt_list: List[Any], pred: Any) -> bool:
    """True if the prediction token-overlaps any GT list element."""
    if pred is None:
        return False
    p = _norm(pred)
    if not p:
        return False
    p_toks = set(p.replace("-", " ").split())
    for item in gt_list:
        g = _norm(item)
        if not g:
            continue
        if g == p or g in p or p in g:
            return True
        if p_toks & set(g.replace("-", " ").split()):
            return True
    return False


def field_cell(pf: Mapping[str, Any], gt_raw: Any, pred_raw: Any) -> str:
    """Derive display status, separating real fabrication from GT-list artifacts.

    The frozen scorer's ``normalize_value`` returns None for list-valued GT, so
    those fields get ``gt_positive == 0``. Any model assertion on them is then
    counted as fabrication even when semantically correct (10,173 single-element
    and 1,475 multi-element GT lists in the test set). We surface those as
    ``listhit`` / ``listmiss`` instead of ``hallucination`` so the view
    distinguishes annotation-format artifacts from genuine hallucination.
    """
    gtp = float(pf.get("gt_positive", 0))
    nga = float(pf.get("n_gen_assert", 0))
    stp = float(pf.get("soft_tp", 0))

    gt_is_list = isinstance(gt_raw, list) and any(
        x is not None and str(x).strip() for x in gt_raw
    )
    if gt_is_list and gtp == 0:
        if nga == 0 or pred_raw is None or not str(pred_raw).strip():
            return "listmiss"
        return "listhit" if _list_hit(gt_raw, pred_raw) else "listmiss"

    if gtp == 0 and nga == 0:
        return "both_unknown"
    if gtp == 0 and nga > 0:
        return "hallucination"   # GT genuinely unknown; model asserted
    if gtp > 0 and nga == 0:
        return "miss"
    return "match" if stp / gtp >= 0.8 else "mismatch"


# ---------------------------------------------------------------------------
# Core build
# ---------------------------------------------------------------------------

def build() -> Tuple[List[Dict], Dict]:
    # 1. Image metadata from test.jsonl
    meta_by_id: Dict[str, Dict] = {}
    for row in iter_jsonl(TEST_JSONL):
        sid = str(row.get("sample_id") or row.get("id") or "")
        if sid:
            meta_by_id[sid] = {
                "image":    row.get("image", ""),
                "bbox_xyxy": row.get("bbox_xyxy"),
                "scene":    row.get("scene", ""),
                "gt_caption": row.get("label", ""),
            }

    # 2. Per-candidate scores and extractions
    cand_scores: Dict[str, List] = {}
    cand_extractions: Dict[str, List] = {}
    for cname in CANDIDATES:
        cs = json.loads((EVAL_DIR / cname / "candidate_scores.json").read_text(encoding="utf-8"))
        cand_scores[cname] = cs["lexical"]["samples"]
        cand_extractions[cname] = list(iter_jsonl(EVAL_DIR / cname / "extractions.jsonl"))

    n = len(cand_scores["v4b_sft"])

    # 3. Merge into unified sample list
    all_items: List[Dict] = []
    for idx in range(n):
        sid = cand_scores["v4b_sft"][idx]["sample_id"]
        meta = meta_by_id.get(sid, {})
        if not meta.get("image"):
            continue

        v4b_f1 = cand_scores["v4b_sft"][idx]["f1"]
        t = tier_of(v4b_f1)
        gt_attrs = cand_extractions["v4b_sft"][idx].get("ground_truth") or {}

        cands_out = []
        for cname in CANDIDATES:
            sc  = cand_scores[cname][idx]
            ex  = cand_extractions[cname][idx]
            attrs = ex.get("attributes") or {}
            fields_out = []
            for fld in FIELDS_ORDER:
                pf = sc.get("per_field", {}).get(fld, {})
                gt_val   = _nested(gt_attrs, fld) if isinstance(gt_attrs, Mapping) else None
                pred_val = _nested(attrs, fld)    if isinstance(attrs,    Mapping) else None
                fields_out.append({
                    "field":  fld,
                    "label":  FIELD_LABELS[fld],
                    "gt":     flatten_gt(gt_val),
                    "pred":   "" if pred_val is None else str(pred_val),
                    "status": field_cell(pf, gt_val, pred_val),
                    "stp":    round(float(pf.get("soft_tp", 0)), 3),
                })
            gt_extra   = gt_attrs.get("extra")   if isinstance(gt_attrs, Mapping) else []
            pred_extra = attrs.get("extra")       if isinstance(attrs,    Mapping) else []
            cands_out.append({
                "name":    cname,
                "label":   CANDIDATE_LABELS[cname],
                "caption": ex.get("prediction", ""),
                "f1":      round(sc["f1"], 4),
                "fab":     round(float(sc["counts"].get("n_fab", 0)), 2),
                "fields":  fields_out,
                "gt_extra":   gt_extra   if isinstance(gt_extra,   list) else [],
                "pred_extra": pred_extra if isinstance(pred_extra, list) else [],
            })

        all_items.append({
            "sample_id": sid,
            "f1_v4b":    round(v4b_f1, 4),
            "tier":      t["id"],
            "image":     meta["image"],
            "bbox_xyxy": meta.get("bbox_xyxy"),
            "scene":     meta.get("scene", ""),
            "gt_caption": meta.get("gt_caption", ""),
            "candidates": cands_out,
        })

    # 4. Bin → sample
    buckets: Dict[int, List] = {t["id"]: [] for t in TIERS}
    for it in all_items:
        buckets[it["tier"]].append(it)

    rng = random.Random(SEED)
    sampled: List[Dict] = []
    tier_summary = []
    for t in TIERS:
        pool  = buckets[t["id"]]
        picks = pool if len(pool) <= PER_TIER else rng.sample(pool, PER_TIER)
        picks = sorted(picks, key=lambda x: x["f1_v4b"], reverse=(t["id"] >= 3))
        for it in picks:
            it["i"] = len(sampled)
            sampled.append(it)
        tier_summary.append({
            "id": t["id"], "label": t["label"], "stars": t["stars"],
            "lo": t["lo"],  "hi": t["hi"],
            "total": len(pool), "sampled": len(picks),
        })

    # 5. Global metrics summary
    global_metrics = {}
    for cname in CANDIDATES:
        cs = json.loads((EVAL_DIR / cname / "candidate_scores.json").read_text(encoding="utf-8"))
        global_metrics[cname] = {
            "label": CANDIDATE_LABELS[cname],
            "micro": cs["lexical"]["metrics"]["micro"],
            "macro_field_f1": cs["lexical"]["metrics"]["macro_field_f1"],
            "fabrication_ratio": cs["lexical"]["metrics"]["fabrication_ratio"],
        }

    meta_out = {
        "tiers":            tier_summary,
        "total_samples":    len(all_items),
        "candidates":       CANDIDATES,
        "candidate_labels": CANDIDATE_LABELS,
        "metrics":          global_metrics,
    }
    return sampled, meta_out


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Florence RL 对比 Demo · V4B / qwen_rl / lexical_rl</title>
<style>
:root{
  --bg:#0f1115;--panel:#171a21;--panel2:#1e2330;--ink:#e6e9ef;--mut:#8b93a4;
  --line:#2a3140;--accent:#6ea8fe;--ok:#3dd68c;--bad:#f7768e;--warn:#ffb454;
  --unk:#454c5e;--hall:#e0af68;--lhit:#2a9d6f;--lmiss:#a05561;
  --t5:#3dd68c;--t4:#9ece6a;--t3:#e0af68;--t2:#ff9e64;--t1:#f7768e;
  --cv4:#6ea8fe;--cqw:#bb9af7;--clex:#73daca;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  font-size:13px;line-height:1.5}
.wrap{max-width:1560px;margin:0 auto;padding:22px 18px 80px}
header{padding:14px 0 10px;border-bottom:1px solid var(--line);margin-bottom:16px}
header h1{font-size:20px;font-weight:650;margin-bottom:3px}
header .sub{color:var(--mut);font-size:12px}
h2.sec{font-size:15px;font-weight:600;margin:24px 0 10px;padding-bottom:6px;
  border-bottom:1px solid var(--line);display:flex;align-items:center;gap:8px}
h2.sec .pill{font-size:11px;background:var(--panel2);color:var(--mut);
  padding:2px 8px;border-radius:10px;font-weight:400}
/* global metrics table */
.gtable{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:4px}
.gtable th,.gtable td{padding:5px 10px;border:1px solid var(--line);text-align:right}
.gtable th{background:var(--panel2);color:var(--mut);font-weight:500;text-align:center}
.gtable td:first-child{text-align:left;color:var(--mut)}
.cv4{color:var(--cv4)}.cqw{color:var(--cqw)}.clex{color:var(--clex)}
/* tier nav */
.tnav{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 4px;
  position:sticky;top:0;background:var(--bg);padding:7px 0;z-index:10}
.tnav a{text-decoration:none;font-size:11px;padding:4px 10px;border-radius:12px;
  border:1px solid var(--line);color:var(--ink);background:var(--panel)}
.tnav a:hover{border-color:var(--accent)}
/* tier heading */
.thead{display:flex;align-items:center;gap:10px;margin:26px 0 10px}
.thead .badge{font-size:12px;font-weight:650;padding:3px 10px;border-radius:7px;color:#0f1115}
.thead .info{font-size:11px;color:var(--mut)}
.thead .cnt{font-size:11px;color:var(--mut);margin-left:auto}
/* card grid */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(680px,1fr));gap:14px}
/* single card */
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.imgrow{position:relative;background:#000;max-height:280px;overflow:hidden}
.imgrow img{width:100%;max-height:280px;object-fit:contain;display:block}
.f1tag{position:absolute;top:7px;left:8px;font-size:12px;font-weight:700;
  padding:2px 8px;border-radius:5px;color:#0f1115}
.sidtag{position:absolute;bottom:5px;left:8px;font-size:10px;color:#cfd3dc;
  background:rgba(0,0,0,.55);padding:2px 6px;border-radius:4px;
  max-width:88%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* body */
.cbody{padding:10px 12px 12px;display:flex;flex-direction:column;gap:8px}
/* GT caption */
.gtcap{font-size:11.5px;color:var(--mut);border-left:3px solid var(--line);
  padding-left:7px;line-height:1.4}
.gtcap .lab{font-size:10px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px}
/* 3-col candidate section */
.ccols{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.ccol{background:var(--panel2);border-radius:8px;padding:8px 9px;display:flex;flex-direction:column;gap:6px}
.chead{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600}
.chead .f1b{font-size:10px;background:rgba(0,0,0,.35);padding:1px 6px;border-radius:4px;font-weight:500}
.capline{font-size:12px;line-height:1.4;color:#cfe1ff;border-left:3px solid;padding-left:6px}
.capline.cv4{border-color:var(--cv4)}
.capline.cqw{border-color:var(--cqw)}
.capline.clex{border-color:var(--clex)}
/* attribute table */
.atbl{border-collapse:collapse;width:100%;font-size:11px}
.atbl th,.atbl td{padding:2px 4px;border-bottom:1px solid var(--line);vertical-align:middle}
.atbl th{color:var(--mut);font-weight:500;font-size:10px;background:var(--panel)}
.atbl td.fl{color:var(--mut);white-space:nowrap;font-size:10px}
.atbl td.gtv{color:#b0bac8}
.atbl td.pv{font-size:11px}
.match   .pv{color:var(--ok)}
.mismatch .pv{color:var(--bad)}
.miss     .pv{color:var(--warn);font-style:italic}
.hallucination .pv{color:var(--hall)}
.both_unknown .pv,.both_unknown .gtv{color:var(--unk)}
.listhit  .pv{color:var(--lhit)}
.listmiss .pv{color:var(--lmiss)}
.listhit  .gtv,.listmiss .gtv{color:#8f7fb0}
.listhit  .fl::after,.listmiss .fl::after{content:" ⌗";color:#8f7fb0;font-size:9px}
/* color-coded tier */
.tcolor-5{background:var(--t5)}.tcolor-4{background:var(--t4)}
.tcolor-3{background:var(--t3)}.tcolor-2{background:var(--t2)}
.tcolor-1{background:var(--t1)}
.border-cv4{border-color:var(--cv4)!important}
.border-cqw{border-color:var(--cqw)!important}
.border-clex{border-color:var(--clex)!important}
footer{color:var(--mut);font-size:11px;margin-top:30px;border-top:1px solid var(--line);padding-top:10px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Florence Person Attribute · RL 三路对比 Demo</h1>
  <div class="sub">
    V4B SFT（基线）vs qwen_rl vs lexical_rl · 冻结测试集 4,328 条 · 按 V4B 词法 soft-F1 分 5 挡，每挡随机抽取 20 张
    · 属性由 Qwen3.6-27B-FP8 从 Florence 推理 caption 抽取，与 GT 原始属性（test.jsonl）对比
  </div>
</header>

<h2 class="sec">全局核心指标 <span class="pill">词法 soft-F1</span></h2>
<table class="gtable">
  <thead><tr><th>指标</th>
    <th class="cv4">V4B SFT（基线）</th>
    <th class="cqw">qwen_rl</th>
    <th class="clex">lexical_rl</th>
  </tr></thead>
  <tbody id="gmetrics"></tbody>
</table>

<div class="tnav" id="tnav"></div>
<div id="tiers"></div>

<footer>
  <div style="margin-bottom:6px">颜色规则（属性字段）：
  <span style="color:var(--ok)">绿=匹配（soft-tp/gt≥80%）</span>·
  <span style="color:var(--bad)">红=不匹配</span>·
  <span style="color:var(--warn)">橙=模型漏标</span>·
  <span style="color:var(--hall)">黄=模型多报（GT确为unknown，真幻觉）</span>·
  <span style="color:var(--unk)">灰=双方均unknown</span>·
  <span style="color:var(--lhit)">深绿 ⌗=GT为列表值且模型命中</span>·
  <span style="color:var(--lmiss)">暗红 ⌗=GT为列表值且模型未命中</span>。</div>
  <div style="margin-bottom:6px;color:#8f7fb0">
  ⌗ 标记说明：冻结打分器的 <code>normalize_value</code> 对列表型 GT 返回 None，导致该字段 <code>gt_positive=0</code>，
  模型只要作答就被计入 fabrication —— 即使语义完全正确。全测试集共 10,173 个单元素列表（如 <code>['brown']</code>）
  与 1,475 个多元素列表（如 <code>['black','white']</code> 双色衣物）受此影响。
  这类格子在此单列 ⌗ 状态，不与真幻觉混淆；三候选受影响程度一致，故不影响对比结论，但幻觉率绝对值被系统性高估。
  上方全局指标仍取自冻结产物，未作修正。</div>
  GT属性来自 test.jsonl 原始标注；预测属性由 Qwen3.6-27B-FP8 从 Florence caption 抽取（三候选各自抽取）。
</footer>
</div>

<script>
const DATA = __SAMPLES_JSON__;
const META = __META_JSON__;

function esc(s){
  return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function pct(x){return (x*100).toFixed(2)+'%';}

// Global metrics table
(function(){
  const m = META.metrics;
  const rows = [
    ['Micro F1',      x=>x.micro.f1.toFixed(4)],
    ['Micro P',       x=>x.micro.precision.toFixed(4)],
    ['Micro R',       x=>x.micro.recall.toFixed(4)],
    ['Macro-field F1',x=>x.macro_field_f1.toFixed(4)],
    ['幻觉率',        x=>pct(x.fabrication_ratio)],
  ];
  const cands = META.candidates;
  document.getElementById('gmetrics').innerHTML = rows.map(([label, fn]) =>
    `<tr><td class="fl">${label}</td>${cands.map((c,i)=>{
      const cls = ['cv4','cqw','clex'][i];
      return `<td class="${cls}">${fn(m[c])}</td>`;
    }).join('')}</tr>`
  ).join('');
})();

// Tier nav
document.getElementById('tnav').innerHTML = META.tiers.map(t =>
  `<a href="#tier-${t.id}">${t.stars} 挡位${t.id} <span style="color:var(--mut)">(${t.sampled}/${t.total})</span></a>`
).join('');

// Tier cards
(function(){
  const byTier = {};
  DATA.forEach(s => { (byTier[s.tier] = byTier[s.tier]||[]).push(s); });
  const CCLASS = ['cv4','cqw','clex'];
  let html = '';

  META.tiers.forEach(t => {
    const items = byTier[t.id] || [];
    html += `<div id="tier-${t.id}">`;
    html += `<div class="thead">
      <span class="badge tcolor-${t.id}">${t.stars} 挡位${t.id}</span>
      <span class="info">V4B F1 ∈ ${t.lo.toFixed(1)}–${t.hi===1.01?'1.0':t.hi.toFixed(1)} · ${t.label.split('·')[1].trim()}</span>
      <span class="cnt">共 ${t.total} 条 · 展示 ${items.length}</span>
    </div><div class="grid">`;

    items.forEach(s => {
      const v4b = s.candidates[0], qrl = s.candidates[1], lrl = s.candidates[2];

      // Attribute table rows (18 fields + extra)
      const fieldRows = s.candidates[0].fields.map((fbase, fi) => {
        const preds = s.candidates.map(c => c.fields[fi]);
        return `<tr class="${fbase.status}">
          <td class="fl">${esc(fbase.label)}</td>
          <td class="gtv">${esc(fbase.gt||'—')}</td>
          ${preds.map((p,ci) => `<td class="pv ${CCLASS[ci]}-col" data-st="${p.status}"
            style="color:${statusColor(p.status)}">${esc(p.pred||'—')}</td>`).join('')}
        </tr>`;
      }).join('');

      const extraRow = `<tr>
        <td class="fl" style="color:var(--mut)">extra</td>
        <td class="gtv">${esc((v4b.gt_extra||[]).join('; ')||'—')}</td>
        ${s.candidates.map((_,ci) => `<td class="pv">${esc((s.candidates[ci].pred_extra||[]).join('; ')||'—')}</td>`).join('')}
      </tr>`;

      html += `<div class="card">
        <div class="imgrow">
          <img loading="lazy" src="/rl/img/${s.i}" alt="">
          <span class="f1tag tcolor-${s.tier}">F1 ${s.f1_v4b.toFixed(3)}</span>
          <span class="sidtag">${esc(s.sample_id)}</span>
        </div>
        <div class="cbody">
          <div class="gtcap"><div class="lab">GT caption</div>${esc(s.gt_caption)}</div>
          <div class="ccols">
            ${s.candidates.map((c,ci)=>`
            <div class="ccol">
              <div class="chead ${CCLASS[ci]}">
                ${esc(c.label)}
                <span class="f1b">F1 ${c.f1.toFixed(3)}</span>
                ${c.fab > 0 ? `<span class="f1b" style="color:var(--bad)">fab ${c.fab}</span>` : ''}
              </div>
              <div class="capline ${CCLASS[ci]}">${esc(c.caption)}</div>
            </div>`).join('')}
          </div>
          <table class="atbl">
            <tr><th>字段</th><th>GT</th>
              <th class="cv4">V4B SFT</th>
              <th class="cqw">qwen_rl</th>
              <th class="clex">lexical_rl</th>
            </tr>
            ${fieldRows}${extraRow}
          </table>
        </div>
      </div>`;
    });
    html += '</div></div>';
  });
  document.getElementById('tiers').innerHTML = html;
})();

function statusColor(st){
  return {
    match:'var(--ok)', mismatch:'var(--bad)', miss:'var(--warn)',
    hallucination:'var(--hall)', both_unknown:'var(--unk)',
    listhit:'var(--lhit)', listmiss:'var(--lmiss)'
  }[st] || 'var(--ink)';
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "rl_build")
    args = parser.parse_args()

    print("Loading data …")
    samples, meta = build()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    server_samples = [{"image": s["image"], "bbox_xyxy": s["bbox_xyxy"]} for s in samples]
    (args.out_dir / "samples.json").write_text(
        json.dumps(server_samples, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    doc = (HTML
           .replace("__SAMPLES_JSON__", json.dumps(samples, ensure_ascii=False))
           .replace("__META_JSON__",    json.dumps(meta,    ensure_ascii=False)))
    (args.out_dir / "index.html").write_text(doc, encoding="utf-8")

    print(f"Built {len(samples)} cards across {len(meta['tiers'])} tiers → {args.out_dir}")
    for t in meta["tiers"]:
        print(f"  挡位{t['id']} {t['stars']}: total={t['total']:4d}  sampled={t['sampled']}")


if __name__ == "__main__":
    main()
