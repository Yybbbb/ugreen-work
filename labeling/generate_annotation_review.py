#!/usr/bin/env python3
"""
生成标注审核网页：从annotation目录均匀抽取100个样本，
嵌入图片，让标注人员选择upper/lower颜色，最终计算与Qwen的比较指标。
"""

import json
import os
import random
import base64
import glob
from pathlib import Path


ANNOTATION_DIR = Path("/data1/work/MichaelYu/segment-color/labeling/annotation")
OUTPUT_HTML = Path("/data1/work/MichaelYu/segment-color/labeling/annotation_review.html")
N_SAMPLES = 100

COLORS = [
    {"english": "unknown",  "chinese": "未知",  "rgb": None},
    {"english": "black",    "chinese": "黑色",  "rgb": [0, 0, 0]},
    {"english": "white",    "chinese": "白色",  "rgb": [255, 255, 255]},
    {"english": "gray",     "chinese": "灰色",  "rgb": [128, 128, 128]},
    {"english": "red",      "chinese": "红色",  "rgb": [255, 0, 0]},
    {"english": "yellow",   "chinese": "黄色",  "rgb": [255, 255, 0]},
    {"english": "green",    "chinese": "绿色",  "rgb": [0, 200, 0]},
    {"english": "blue",     "chinese": "蓝色",  "rgb": [0, 0, 255]},
    {"english": "purple",   "chinese": "紫色",  "rgb": [128, 0, 128]},
    {"english": "pink",     "chinese": "粉色",  "rgb": [255, 192, 203]},
    {"english": "orange",   "chinese": "橙色",  "rgb": [255, 165, 0]},
    {"english": "brown",    "chinese": "棕色",  "rgb": [137, 81, 41]},
]


def uniform_sample(all_files, n):
    """均匀间隔抽样"""
    if len(all_files) <= n:
        return all_files
    step = len(all_files) / n
    return [all_files[int(i * step)] for i in range(n)]


def load_image_base64(image_path):
    """将图片读取为base64字符串"""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        ext = Path(image_path).suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"  [WARN] 无法读取图片 {image_path}: {e}")
        return None


def get_qwen_colors(parsed, part):
    """返回 Qwen 对某部位的颜色列表（按置信度从高到低），最多3个"""
    colors = parsed.get(part, [])
    if not colors:
        return ["unknown"]
    sorted_colors = sorted(colors, key=lambda x: x.get("confidence", 0), reverse=True)
    return [c["label"] for c in sorted_colors[:3]]


def build_samples(annotation_files):
    """加载并构建样本数据"""
    samples = []
    skipped = 0
    for fpath in annotation_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] 解析失败 {fpath}: {e}")
            skipped += 1
            continue

        image_path = data.get("image_path", "")
        parsed = data.get("parsed", {})
        status = data.get("status", "")

        if status != "success" or not parsed:
            skipped += 1
            continue

        qwen_upper_all = get_qwen_colors(parsed, "upper")
        qwen_lower_all = get_qwen_colors(parsed, "lower")

        img_b64 = load_image_base64(image_path)
        if img_b64 is None:
            skipped += 1
            continue

        samples.append({
            "id": Path(fpath).stem,
            "image_path": image_path,
            "image_b64": img_b64,
            "qwen_upper": qwen_upper_all[0],       # 主要颜色（top-1）
            "qwen_upper_all": qwen_upper_all,       # 全部颜色（最多3个）
            "qwen_lower": qwen_lower_all[0],
            "qwen_lower_all": qwen_lower_all,
        })

    print(f"成功加载 {len(samples)} 个样本，跳过 {skipped} 个")
    return samples


def generate_html(samples):
    colors_js = json.dumps(COLORS, ensure_ascii=False)
    samples_js = json.dumps(
        [{"id": s["id"],
          "image_path": s["image_path"],
          "image": s["image_b64"],
          "qwen_upper": s["qwen_upper"],
          "qwen_upper_all": s["qwen_upper_all"],
          "qwen_lower": s["qwen_lower"],
          "qwen_lower_all": s["qwen_lower_all"]}
         for s in samples],
        ensure_ascii=False
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>服装颜色标注审核</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "PingFang SC", "Microsoft YaHei", sans-serif; background: #f0f2f5; color: #333; }}

  /* ====== 顶部进度栏 ====== */
  #header {{
    position: sticky; top: 0; z-index: 100;
    background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,.12);
    padding: 12px 24px; display: flex; align-items: center; gap: 16px;
  }}
  #header h1 {{ font-size: 18px; flex: 1; }}
  #progress-text {{ font-size: 14px; color: #666; white-space: nowrap; }}
  #progress-bar-wrap {{ flex: 1; background: #e0e0e0; border-radius: 6px; height: 8px; min-width: 120px; }}
  #progress-bar {{ height: 8px; border-radius: 6px; background: #1890ff; transition: width .3s; width: 0%; }}

  /* ====== 主体 ====== */
  #main {{ max-width: 900px; margin: 24px auto; padding: 0 16px; }}

  /* ====== 卡片 ====== */
  .card {{
    background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08);
    padding: 24px; margin-bottom: 24px;
  }}
  .card-title {{ font-size: 15px; color: #888; margin-bottom: 12px; }}

  /* ====== 图片区 ====== */
  #img-wrap {{ text-align: center; }}
  #img-wrap img {{ max-height: 420px; max-width: 100%; border-radius: 8px; object-fit: contain; }}
  #sample-id {{ margin-top: 6px; font-size: 12px; color: #bbb; }}

  /* ====== 颜色选择 ====== */
  .color-section {{ margin-top: 20px; }}
  .color-section h3 {{ font-size: 15px; margin-bottom: 12px; }}
  .color-grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .color-btn {{
    display: flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 20px; border: 2px solid #e0e0e0;
    background: #fafafa; cursor: pointer; font-size: 13px; transition: all .15s;
    user-select: none;
  }}
  .color-btn:hover {{ border-color: #1890ff; background: #e6f4ff; }}
  .color-btn.selected {{ border-color: #1890ff; background: #1890ff; color: #fff; }}
  .color-btn.selected .swatch {{ border-color: #fff; }}
  .swatch {{
    width: 16px; height: 16px; border-radius: 50%; border: 1px solid #ccc; flex-shrink: 0;
  }}
  .unknown-swatch {{ background: repeating-linear-gradient(45deg,#ccc,#ccc 2px,#fff 2px,#fff 6px); }}

  /* ====== 导航按钮 ====== */
  #nav {{ display: flex; justify-content: space-between; align-items: center; margin-top: 24px; }}
  .btn {{
    padding: 10px 28px; border-radius: 8px; border: none; cursor: pointer;
    font-size: 15px; font-weight: 500; transition: all .15s;
  }}
  .btn-primary {{ background: #1890ff; color: #fff; }}
  .btn-primary:hover {{ background: #0d7de6; }}
  .btn-primary:disabled {{ background: #b0d4f5; cursor: not-allowed; }}
  .btn-secondary {{ background: #f0f0f0; color: #555; }}
  .btn-secondary:hover {{ background: #e0e0e0; }}
  .btn-secondary:disabled {{ opacity: .4; cursor: not-allowed; }}

  /* ====== 完成提示 ====== */
  #done-notice {{
    text-align: center; padding: 40px 20px; display: none;
  }}
  #done-notice h2 {{ font-size: 22px; margin-bottom: 8px; }}
  #done-notice p {{ color: #666; margin-bottom: 20px; }}

  /* ====== 结果面板 ====== */
  #result-panel {{ display: none; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }}
  .metric-card {{
    background: #f7f9ff; border-radius: 10px; padding: 16px; text-align: center;
    border: 1px solid #d6e4ff;
  }}
  .metric-card .val {{ font-size: 28px; font-weight: 700; color: #1890ff; }}
  .metric-card .label {{ font-size: 12px; color: #888; margin-top: 4px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f0f4ff; padding: 10px 8px; text-align: center; border-bottom: 2px solid #d0ddf0; }}
  td {{ padding: 8px; text-align: center; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f7f9ff; }}
  .match {{ color: #52c41a; font-weight: 600; }}
  .match3 {{ color: #fa8c16; font-weight: 600; }}
  .mismatch {{ color: #ff4d4f; font-weight: 600; }}

  .per-class-table td, .per-class-table th {{ text-align: center; }}
</style>
</head>
<body>

<div id="header">
  <h1>服装颜色标注审核</h1>
  <span id="progress-text">0 / {len(samples)}</span>
  <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
  <button class="btn btn-secondary" id="btn-show-result" style="display:none" onclick="showResult()">查看结果</button>
</div>

<div id="main">
  <!-- 审核说明 -->
  <div class="card" style="background:#f0f7ff;border:1px solid #91caff;margin-bottom:16px;">
    <div style="font-size:15px;font-weight:600;margin-bottom:10px;color:#0958d9">📋 标注说明</div>
    <ul style="font-size:14px;color:#333;line-height:2;padding-left:20px;">
      <li>请根据图片中人物的<b>上衣</b>和<b>下装</b>分别选择一个颜色。</li>
      <li><b>「未知」</b>选项仅用于以下两种情况：图片中人物<b>没有穿</b>该部位的衣物，或者<b>完全看不到</b>该部位。</li>
      <li>只要能看到衣物，请尽可能选择实际的颜色，不要选未知。</li>
      <li>按照<b>第一感觉</b>选择，不需要过分推理，快速判断即可。</li>
      <li>中途可以随时关闭页面，进度会自动保存，下次打开同一链接可继续标注。</li>
      <li style="color:#d4380d;font-weight:600">⚠️ 全部标注完成后，必须点击「💾 保存结果到服务器」，否则结果不会写入磁盘。</li>
    </ul>
  </div>

  <!-- 标注卡片 -->
  <div id="annotation-panel">
    <div class="card">
      <div class="card-title" id="card-title">请标注图片中的服装颜色</div>
      <div id="img-wrap">
        <img id="main-img" src="" alt="样本图片">
        <div id="sample-id"></div>
      </div>
      <div class="color-section">
        <h3>上衣（Upper）颜色</h3>
        <div class="color-grid" id="upper-colors"></div>
      </div>
      <div class="color-section">
        <h3>下装（Lower）颜色</h3>
        <div class="color-grid" id="lower-colors"></div>
      </div>
    </div>

    <div id="nav">
      <button class="btn btn-secondary" id="btn-prev" onclick="navigate(-1)" disabled>← 上一张</button>
      <span id="nav-label" style="color:#888;font-size:14px"></span>
      <button class="btn btn-primary" id="btn-next" onclick="navigate(1)" disabled>下一张 →</button>
    </div>
  </div>

  <!-- 完成提示 -->
  <div id="done-notice" class="card">
    <h2>🎉 标注完成！</h2>
    <p style="margin-bottom:16px">您已完成全部 {len(samples)} 张图片的标注。</p>
    <div style="background:#fff7e6;border:1px solid #ffa940;border-radius:8px;padding:14px 18px;margin-bottom:20px;font-size:14px;">
      ⚠️ <strong>请先点击「保存结果到服务器」</strong>，确保结果已写入磁盘后再查看比较报告。
    </div>
    <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
      <button class="btn btn-primary" id="btn-save-server" onclick="saveToServer()">💾 保存结果到服务器</button>
      <button class="btn btn-secondary" onclick="showResult()">查看比较结果</button>
    </div>
    <div id="save-status" style="margin-top:12px;font-size:13px;text-align:center;"></div>
  </div>

  <!-- 结果面板 -->
  <div id="result-panel" class="card">
    <h2 style="margin-bottom:16px">标注结果 vs AI（Qwen）</h2>
    <div id="metrics-area"></div>
    <h3 style="margin:20px 0 10px">逐样本对比</h3>
    <div style="overflow-x:auto">
      <table id="detail-table">
        <thead>
          <tr>
            <th>#</th><th>样本ID</th>
            <th>Upper（人工）</th><th>Upper（AI Top-1）</th><th>匹配</th>
            <th>Lower（人工）</th><th>Lower（AI Top-1）</th><th>匹配</th>
          </tr>
          <tr style="font-size:11px;color:#888">
            <td colspan="4"></td><td>✓=Top-1 ≈=Top-3 ✗=不符</td>
            <td colspan="2"></td><td>✓=Top-1 ≈=Top-3 ✗=不符</td>
          </tr>
        </thead>
        <tbody id="detail-tbody"></tbody>
      </table>
    </div>
    <div style="margin-top:24px">
      <h3 style="margin-bottom:10px">各颜色类别统计（Upper + Lower 合并）</h3>
      <div style="overflow-x:auto">
        <table class="per-class-table" id="class-table">
          <thead>
            <tr>
              <th rowspan="2">颜色</th>
              <th rowspan="2">AI预测数<br>(Top-1)</th>
              <th rowspan="2">人工标注数</th>
              <th colspan="4" style="background:#e6f7e6">Top-1 一致</th>
              <th colspan="4" style="background:#fff7e6">Top-3 一致</th>
            </tr>
            <tr>
              <th style="background:#e6f7e6">TP</th>
              <th style="background:#e6f7e6">Precision</th>
              <th style="background:#e6f7e6">Recall</th>
              <th style="background:#e6f7e6">F1</th>
              <th style="background:#fff7e6">TP</th>
              <th style="background:#fff7e6">Precision</th>
              <th style="background:#fff7e6">Recall</th>
              <th style="background:#fff7e6">F1</th>
            </tr>
          </thead>
          <tbody id="class-tbody"></tbody>
        </table>
      </div>
    </div>
    <div style="margin-top:24px;display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
      <button class="btn btn-primary" id="btn-save-server2" onclick="saveToServer()">💾 保存结果到服务器</button>
      <button class="btn btn-secondary" onclick="downloadResult()">下载结果 JSON</button>
    </div>
    <div id="save-status2" style="margin-top:12px;font-size:13px;text-align:center;"></div>
  </div>
</div>

<script>
const COLORS = {colors_js};
const SAMPLES = {samples_js};

const STORAGE_KEY = 'annotation_review_v1';

// 从 localStorage 恢复进度
let current = 0;
let annotations = {{}};

function loadFromStorage() {{
  try {{
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {{
      const data = JSON.parse(saved);
      annotations = data.annotations || {{}};
      current = Math.min(data.current || 0, SAMPLES.length - 1);
      return true;
    }}
  }} catch(e) {{}}
  return false;
}}

function saveToStorage() {{
  try {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify({{ annotations, current }}));
  }} catch(e) {{}}
}}

function clearStorage() {{
  localStorage.removeItem(STORAGE_KEY);
}}

function swatchStyle(color) {{
  if (!color.rgb) return 'background:transparent;';
  return `background:rgb(${{color.rgb.join(',')}});`;
}}

function buildColorButtons(containerId, part, sampleId) {{
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  COLORS.forEach(c => {{
    const btn = document.createElement('div');
    btn.className = 'color-btn';
    btn.dataset.color = c.english;
    btn.dataset.part = part;

    const swatch = document.createElement('span');
    swatch.className = 'swatch' + (c.rgb ? '' : ' unknown-swatch');
    if (c.rgb) swatch.style.cssText = swatchStyle(c);

    btn.appendChild(swatch);
    btn.appendChild(document.createTextNode(c.chinese));
    btn.onclick = () => selectColor(sampleId, part, c.english, containerId);
    container.appendChild(btn);
  }});
}}

function selectColor(sampleId, part, color, containerId) {{
  if (!annotations[sampleId]) annotations[sampleId] = {{}};
  annotations[sampleId][part] = color;

  // 每次选择后立即保存到 localStorage
  saveToStorage();

  document.querySelectorAll(`#${{containerId}} .color-btn`).forEach(b => {{
    b.classList.toggle('selected', b.dataset.color === color);
  }});

  checkNextEnabled();
}}

function checkNextEnabled() {{
  const s = SAMPLES[current];
  const ann = annotations[s.id] || {{}};
  const ready = ann.upper && ann.lower;
  document.getElementById('btn-next').disabled = !ready;
}}

function renderSample(idx) {{
  const s = SAMPLES[idx];
  document.getElementById('main-img').src = s.image;
  document.getElementById('sample-id').textContent = `ID: ${{s.id}}`;
  document.getElementById('card-title').textContent = `图片 ${{idx + 1}} / ${{SAMPLES.length}}`;
  document.getElementById('nav-label').textContent = `${{idx + 1}} / ${{SAMPLES.length}}`;

  // 更新进度
  const pct = Math.round(idx / SAMPLES.length * 100);
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = `${{idx}} / ${{SAMPLES.length}}`;

  buildColorButtons('upper-colors', 'upper', s.id);
  buildColorButtons('lower-colors', 'lower', s.id);

  // 恢复已选择的颜色
  const ann = annotations[s.id] || {{}};
  if (ann.upper) {{
    document.querySelectorAll('#upper-colors .color-btn').forEach(b => {{
      b.classList.toggle('selected', b.dataset.color === ann.upper);
    }});
  }}
  if (ann.lower) {{
    document.querySelectorAll('#lower-colors .color-btn').forEach(b => {{
      b.classList.toggle('selected', b.dataset.color === ann.lower);
    }});
  }}

  document.getElementById('btn-prev').disabled = idx === 0;
  checkNextEnabled();
}}

function navigate(dir) {{
  const next = current + dir;
  if (next < 0 || next >= SAMPLES.length) return;

  if (next >= SAMPLES.length) {{
    finishAnnotation();
    return;
  }}
  current = next;

  if (current === SAMPLES.length - 1) {{
    document.getElementById('btn-next').textContent = '完成 ✓';
  }} else {{
    document.getElementById('btn-next').textContent = '下一张 →';
  }}

  // 若是最后一张点"完成"
  if (dir === 1 && current === SAMPLES.length) {{
    finishAnnotation();
    return;
  }}

  renderSample(current);
}}

// 点击"下一张"时若已是最后一张则完成
document.getElementById('btn-next').addEventListener('click', function() {{
  const ann = annotations[SAMPLES[current].id] || {{}};
  if (!ann.upper || !ann.lower) return;

  if (current === SAMPLES.length - 1) {{
    finishAnnotation();
  }} else {{
    current++;
    if (current === SAMPLES.length - 1) {{
      document.getElementById('btn-next').textContent = '完成 ✓';
    }}
    renderSample(current);
  }}
}}, true);

// 覆盖 navigate 的点击，避免重复绑定；移除原有 onclick
document.getElementById('btn-next').removeAttribute('onclick');

function finishAnnotation() {{
  document.getElementById('annotation-panel').style.display = 'none';
  document.getElementById('done-notice').style.display = 'block';
  document.getElementById('progress-bar').style.width = '100%';
  document.getElementById('progress-text').textContent = `${{SAMPLES.length}} / ${{SAMPLES.length}}`;
  document.getElementById('btn-show-result').style.display = '';
}}

function calcMetrics() {{
  let upper_tp1 = 0, upper_tp3 = 0, upper_total = 0;
  let lower_tp1 = 0, lower_tp3 = 0, lower_total = 0;
  const details = [];

  // 按颜色统计（合并upper/lower），分别统计top1和top3
  const colorStats = {{}};
  COLORS.forEach(c => {{
    colorStats[c.english] = {{ai_pred: 0, human_pred: 0, tp1: 0, tp3: 0}};
  }});

  SAMPLES.forEach(s => {{
    const ann = annotations[s.id] || {{}};
    const hu = ann.upper || '(未标注)';
    const hl = ann.lower || '(未标注)';
    const au1 = s.qwen_upper;
    const al1 = s.qwen_lower;
    const au_all = s.qwen_upper_all || [au1];
    const al_all = s.qwen_lower_all || [al1];

    const u_match1 = hu === au1;
    const l_match1 = hl === al1;
    const u_match3 = au_all.includes(hu);
    const l_match3 = al_all.includes(hl);

    if (ann.upper) {{
      upper_total++;
      if (u_match1) upper_tp1++;
      if (u_match3) upper_tp3++;
    }}
    if (ann.lower) {{
      lower_total++;
      if (l_match1) lower_tp1++;
      if (l_match3) lower_tp3++;
    }}

    details.push({{ id: s.id, image_path: s.image_path,
      hu, hl, au1, al1, au_all, al_all,
      u_match1, l_match1, u_match3, l_match3 }});

    // 颜色级别统计（用top-1作为ai_pred基准）
    if (ann.upper) {{
      if (colorStats[au1] !== undefined) colorStats[au1].ai_pred++;
      if (colorStats[hu] !== undefined) colorStats[hu].human_pred++;
      if (u_match1 && colorStats[au1] !== undefined) colorStats[au1].tp1++;
      if (u_match3 && colorStats[au1] !== undefined) colorStats[au1].tp3++;
    }}
    if (ann.lower) {{
      if (colorStats[al1] !== undefined) colorStats[al1].ai_pred++;
      if (colorStats[hl] !== undefined) colorStats[hl].human_pred++;
      if (l_match1 && colorStats[al1] !== undefined) colorStats[al1].tp1++;
      if (l_match3 && colorStats[al1] !== undefined) colorStats[al1].tp3++;
    }}
  }});

  const fmt = (n, d) => d ? (n / d * 100).toFixed(1) : 'N/A';
  return {{
    upper_acc1: fmt(upper_tp1, upper_total),
    upper_acc3: fmt(upper_tp3, upper_total),
    lower_acc1: fmt(lower_tp1, lower_total),
    lower_acc3: fmt(lower_tp3, lower_total),
    overall_acc1: fmt(upper_tp1 + lower_tp1, upper_total + lower_total),
    overall_acc3: fmt(upper_tp3 + lower_tp3, upper_total + lower_total),
    upper_tp1, upper_tp3, lower_tp1, lower_tp3,
    upper_total, lower_total,
    details, colorStats
  }};
}}

function showResult() {{
  document.getElementById('done-notice').style.display = 'none';
  document.getElementById('annotation-panel').style.display = 'none';
  document.getElementById('result-panel').style.display = 'block';

  const m = calcMetrics();

  // 总体指标：两行，Top-1 和 Top-3
  document.getElementById('metrics-area').innerHTML = `
    <p style="font-size:13px;color:#888;margin-bottom:8px">
      <b>Top-1 一致</b>：人工选择 = Qwen 最高置信度颜色 &nbsp;|&nbsp;
      <b>Top-3 一致</b>：人工选择出现在 Qwen 所有预测颜色中（≤3个）
    </p>
    <div class="metric-grid">
      <div class="metric-card"><div class="val">${{m.overall_acc1}}%</div><div class="label">整体一致率（Top-1）</div></div>
      <div class="metric-card"><div class="val">${{m.overall_acc3}}%</div><div class="label">整体一致率（Top-3）</div></div>
      <div class="metric-card"><div class="val">${{m.upper_acc1}}%</div><div class="label">Upper（Top-1）</div></div>
      <div class="metric-card"><div class="val">${{m.upper_acc3}}%</div><div class="label">Upper（Top-3）</div></div>
      <div class="metric-card"><div class="val">${{m.lower_acc1}}%</div><div class="label">Lower（Top-1）</div></div>
      <div class="metric-card"><div class="val">${{m.lower_acc3}}%</div><div class="label">Lower（Top-3）</div></div>
    </div>`;

  // 逐样本对比表（含 top-3 列）
  const tbody = document.getElementById('detail-tbody');
  tbody.innerHTML = '';
  m.details.forEach((d, i) => {{
    const qwen_u_str = d.au_all.join(' / ');
    const qwen_l_str = d.al_all.join(' / ');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{i+1}}</td>
      <td style="font-size:11px;word-break:break-all;max-width:140px">${{d.id}}</td>
      <td>${{d.hu}}</td>
      <td title="${{qwen_u_str}}">${{d.au1}}${{d.au_all.length > 1 ? ' <span style="color:#aaa;font-size:11px">+' + (d.au_all.length-1) + '</span>' : ''}}</td>
      <td class="${{d.u_match1 ? 'match' : (d.u_match3 ? 'match3' : 'mismatch')}}">
        ${{d.u_match1 ? '✓' : (d.u_match3 ? '≈' : '✗')}}</td>
      <td>${{d.hl}}</td>
      <td title="${{qwen_l_str}}">${{d.al1}}${{d.al_all.length > 1 ? ' <span style="color:#aaa;font-size:11px">+' + (d.al_all.length-1) + '</span>' : ''}}</td>
      <td class="${{d.l_match1 ? 'match' : (d.l_match3 ? 'match3' : 'mismatch')}}">
        ${{d.l_match1 ? '✓' : (d.l_match3 ? '≈' : '✗')}}</td>`;
    tbody.appendChild(tr);
  }});

  // 颜色级别统计（分 top-1 和 top-3）
  const cbody = document.getElementById('class-tbody');
  cbody.innerHTML = '';
  COLORS.forEach(c => {{
    const s = m.colorStats[c.english];
    if (!s || (s.ai_pred === 0 && s.human_pred === 0)) return;
    const prec1 = s.ai_pred ? (s.tp1 / s.ai_pred * 100).toFixed(1) : '-';
    const rec1  = s.human_pred ? (s.tp1 / s.human_pred * 100).toFixed(1) : '-';
    const f1_1  = (s.ai_pred && s.human_pred) ? (2*s.tp1/(s.ai_pred+s.human_pred)*100).toFixed(1) : '-';
    const prec3 = s.ai_pred ? (s.tp3 / s.ai_pred * 100).toFixed(1) : '-';
    const rec3  = s.human_pred ? (s.tp3 / s.human_pred * 100).toFixed(1) : '-';
    const f1_3  = (s.ai_pred && s.human_pred) ? (2*s.tp3/(s.ai_pred+s.human_pred)*100).toFixed(1) : '-';
    const swStyle = c.rgb
      ? `background:rgb(${{c.rgb.join(',')}});`
      : 'background:repeating-linear-gradient(45deg,#ccc,#ccc 2px,#fff 2px,#fff 6px);';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span style="${{swStyle}}display:inline-block;vertical-align:middle;margin-right:4px;
           border:1px solid #ccc;width:14px;height:14px;border-radius:50%"></span>
           ${{c.chinese}}</td>
      <td>${{s.ai_pred}}</td><td>${{s.human_pred}}</td>
      <td>${{s.tp1}}</td><td>${{prec1}}%</td><td>${{rec1}}%</td><td>${{f1_1}}%</td>
      <td>${{s.tp3}}</td><td>${{prec3}}%</td><td>${{rec3}}%</td><td>${{f1_3}}%</td>`;
    cbody.appendChild(tr);
  }});
}}

function buildResultPayload() {{
  const m = calcMetrics();
  const out = {{
    summary: {{
      overall_agreement_top1: m.overall_acc1 + '%',
      overall_agreement_top3: m.overall_acc3 + '%',
      upper_agreement_top1: m.upper_acc1 + '%',
      upper_agreement_top3: m.upper_acc3 + '%',
      lower_agreement_top1: m.lower_acc1 + '%',
      lower_agreement_top3: m.lower_acc3 + '%',
      upper_labeled: m.upper_total,
      lower_labeled: m.lower_total,
    }},
    per_class: {{}},
    details: m.details.map(d => ({{
      id: d.id,
      image_path: d.image_path,
      human_upper: d.hu,
      qwen_upper_top1: d.au1,
      qwen_upper_all: d.au_all,
      upper_match_top1: d.u_match1,
      upper_match_top3: d.u_match3,
      human_lower: d.hl,
      qwen_lower_top1: d.al1,
      qwen_lower_all: d.al_all,
      lower_match_top1: d.l_match1,
      lower_match_top3: d.l_match3,
    }}))
  }};
  COLORS.forEach(c => {{
    const s = m.colorStats[c.english];
    if (s && (s.ai_pred > 0 || s.human_pred > 0)) {{
      out.per_class[c.english] = s;
    }}
  }});
  return out;
}}

async function saveToServer() {{
  const btns = [document.getElementById('btn-save-server'), document.getElementById('btn-save-server2')];
  const statEls = [document.getElementById('save-status'), document.getElementById('save-status2')];
  const setStatus = (html) => statEls.forEach(el => {{ if(el) el.innerHTML = html; }});
  btns.forEach(b => {{ if(b) {{ b.disabled = true; b.textContent = '保存中...'; }} }});

  try {{
    const payload = buildResultPayload();
    const resp = await fetch('/save', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(payload)
    }});
    const result = await resp.json();
    if (resp.ok) {{
      setStatus(`<span style="color:#52c41a;font-weight:600">✅ 已保存到服务器：${{result.path}}</span>`);
      btns.forEach(b => {{ if(b) {{ b.textContent = '✅ 已保存'; }} }});
    }} else {{
      throw new Error(result.error || '服务器返回错误');
    }}
  }} catch(e) {{
    setStatus(`<span style="color:#ff4d4f">❌ 保存失败：${{e.message}}。请点击「下载结果 JSON」手动保存。</span>`);
    btns.forEach(b => {{ if(b) {{ b.disabled = false; b.textContent = '💾 保存结果到服务器'; }} }});
  }}
}}

function downloadResult() {{
  const out = buildResultPayload();
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'annotation_review_result.json';
  a.click();
}}

// 初始化：尝试恢复上次进度
const resumed = loadFromStorage();
if (resumed && current > 0) {{
  const done = Object.keys(annotations).length;
  const banner = document.createElement('div');
  banner.style.cssText = 'background:#fffbe6;border:1px solid #ffe58f;border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:13px;color:#614700;';
  banner.innerHTML = `⚡ 已自动恢复上次进度，已标注 <b>${{done}}</b> 张（共 ${{SAMPLES.length}} 张）。
    <a href="#" onclick="resetProgress()" style="margin-left:12px;color:#ff4d4f;">重新开始</a>`;
  document.getElementById('main').insertBefore(banner, document.getElementById('annotation-panel'));
}}

// 检查是否已全部完成
const completedCount = SAMPLES.filter(s => annotations[s.id]?.upper && annotations[s.id]?.lower).length;
if (completedCount === SAMPLES.length && SAMPLES.length > 0) {{
  finishAnnotation();
}} else {{
  if (current === SAMPLES.length - 1) {{
    document.getElementById('btn-next').textContent = '完成 ✓';
  }}
  renderSample(current);
}}

function resetProgress() {{
  if (!confirm('确定要清除所有标注进度重新开始吗？')) return;
  clearStorage();
  location.reload();
}}
</script>
</body>
</html>
"""
    return html


def main():
    print("扫描标注文件...")
    all_files = sorted(glob.glob(str(ANNOTATION_DIR / "*.json")))
    print(f"共找到 {len(all_files)} 个文件")

    print(f"均匀抽取 {N_SAMPLES} 个样本...")
    sampled_files = uniform_sample(all_files, N_SAMPLES)

    print("加载样本数据（含图片base64编码，可能需要一分钟）...")
    samples = build_samples(sampled_files)

    if len(samples) < N_SAMPLES:
        print(f"[WARN] 有效样本仅 {len(samples)} 个（目标 {N_SAMPLES}）")

    print("生成HTML...")
    html = generate_html(samples)

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUTPUT_HTML.stat().st_size / 1024 / 1024
    print(f"\n完成！输出文件: {OUTPUT_HTML}")
    print(f"文件大小: {size_mb:.1f} MB")
    print(f"有效样本数: {len(samples)}")
    print("\n用浏览器打开该HTML文件即可开始标注。")


if __name__ == "__main__":
    main()
