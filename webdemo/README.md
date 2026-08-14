# Florence V4A 测试集结果 Web Demo

按 per-sample 严格 Micro F1 把 V4A (replay1k) 的 4,328 条测试结果分成 5 个挡位，每挡随机抽 20 张（不足则全取），在浏览器里对比 **Florence 推理 caption** 与 **GT caption**，以及逐字段属性对比。

**GT 基准为 Qwen-GT**：GT 属性与预测属性都由 `Qwen36-35b-caption` 从各自 caption 抽取（GT 取自 GT caption → `data/prepared/test_qwen_attributes.jsonl`，预测取自 Florence 推理 caption），同口径对比，消除手工标注与 Qwen 抽取的 schema/词汇偏差。指标来自 V4A 的 `metrics_qwen_gt.json`（Micro F1=0.7116、Macro-field F1=0.5864）。

## 结构

```
webdemo/
├── build_demo.py   # 读取 V4A predictions + Qwen 评估，算 per-sample F1，分 5 挡抽样，生成 index.html
├── server.py       # FastAPI：提供 index.html 与 /img/{i}（原图叠红框人物检测框）
├── run_demo.sh     # 一键构建 + 启动
└── build/          # 生成产物（index.html / samples.json / metrics.json）
```

## 启动

```bash
cd /data1/work/MichaelYu/florence-attibute
./webdemo/run_demo.sh
```

打开 `http://127.0.0.1:8012`（或对应 PORT）。

环境变量：

- `PORT` / `HOST`：监听地址，默认 `0.0.0.0:8012`。
- `PYTHON_BIN`：解释器，默认 florence conda 环境。
- `REBUILD=0`：跳过重新构建，直接用已有 `build/index.html`。

## 挡位定义

按 per-sample 严格 Micro F1（复用 `scripts/evaluate_qwen_attribute_extraction.py` 的 `score_attribute_pair`，tp/fp/fn over 已知属性 + extra）等宽分 5 挡：

| 挡位 | F1 区间 | 含义 |
|---|---|---|
| 5 | (0.8, 1.0] | 优秀 |
| 4 | (0.6, 0.8] | 良好 |
| 3 | (0.4, 0.6] | 中等 |
| 2 | (0.2, 0.4] | 较差 |
| 1 | [0, 0.2] | 很差 |

每挡用固定随机种子 `20260807` 抽样，结果可复现。

## 页面内容

- **顶部**：V4A 测试集核心指标（Micro/Soft/Macro F1、Mean field exact、Unknown-extra、平均词数、背景关键词率等）+ 18 字段 F1 条形图。
- **5 个挡位区**：每张卡片含 原图（红框=人物检测框）、Florence 推理 caption、GT caption、逐字段属性表（绿=匹配 / 红=不匹配 / 橙=漏标或多余 / 灰=双方未知）。属性区字体偏小偏密，重点放在 caption 对比上。
