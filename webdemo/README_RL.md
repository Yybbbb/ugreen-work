# RL 三路对比 Demo

对比 V4B SFT（基线）、qwen_rl、lexical_rl 在冻结测试集上的推理结果。
按 V4B 词法 soft-F1 分 5 挡，每挡随机抽取 20 张（seed=20260812）。

## 运行

```bash
cd /data1/work/MichaelYu/florence-attibute
bash webdemo/run_rl_demo.sh
# → http://localhost:8013
```

或分两步：

```bash
python webdemo/build_rl_demo.py          # 生成 webdemo/rl_build/
python webdemo/rl_server.py --port 8013  # 起服务
```

`index.html` 每次请求都重读，改完 build 不用重启服务。

## 数据来源

| 内容 | 来源 |
|---|---|
| 图片路径 / bbox | `data/prepared/test.jsonl` |
| Florence 推理 caption | `artifacts/rl/evaluation/{候选}/extractions.jsonl` 的 `prediction` |
| 预测属性 | 同上的 `attributes`（Qwen3.6-27B-FP8 抽取） |
| GT 属性 | 同上的 `ground_truth`（test.jsonl 原始标注） |
| per-sample F1 / per-field 分数 | `artifacts/rl/evaluation/{候选}/candidate_scores.json` 的 `lexical.samples` |

三个候选的 4,328 行按 sample_id 严格同序，已在构建时校验。

## 挡位分布

| 挡位 | V4B F1 区间 | 池子大小 | 展示 |
|---|---|---:|---:|
| 5 ★★★★★ 优秀 | 0.8–1.0 | 1,110 | 20 |
| 4 ★★★★ 良好 | 0.6–0.8 | 2,162 | 20 |
| 3 ★★★ 中等 | 0.4–0.6 | 913 | 20 |
| 2 ★★ 较差 | 0.2–0.4 | 132 | 20 |
| 1 ★ 很差 | 0.0–0.2 | 11 | 11（全取） |

## 字段颜色

| 颜色 | 状态 | 含义 |
|---|---|---|
| 绿 | `match` | soft-tp / gt_positive ≥ 80% |
| 红 | `mismatch` | 双方都有值但不一致 |
| 橙斜体 | `miss` | GT 有值，模型未提及 |
| 黄 | `hallucination` | GT 确为 unknown，模型编造（真幻觉） |
| 灰 | `both_unknown` | 双方均 unknown，不计分 |
| 深绿 ⌗ | `listhit` | GT 为列表值，模型命中 |
| 暗红 ⌗ | `listmiss` | GT 为列表值，模型未命中 |

## ⌗ 标记：GT 列表值导致的伪幻觉

冻结打分器 `evaluate_qwen_attribute_extraction.normalize_value()` 对**列表型 GT 返回 None**，
于是该字段 `gt_positive=0`；模型只要作答就得 `n_gen_assert=1` → 被计入 `n_fab`（fabrication），
**即使语义完全正确**。

实例：GT `['dark','blue']`，模型输出 `dark blue` —— 语义正确，却记一次幻觉。

全测试集受影响字段：

- 单元素列表 **10,173** 个（如 `['brown']`）
- 多元素列表 **1,475** 个（如 `['black','white']` 双色衣物）

本 demo 91 张卡内的重分类效果：**689 个原"幻觉"格 → 197 个真幻觉 + 280 listhit + 560 listmiss**，
即约 71% 的标黄格子实际是标注格式产物而非模型编造。

影响范围：

- **不影响三路对比结论**。三个候选受影响程度一致（同一套 GT），
  报告里 lexical_rl 的 joint F1 +0.449pp 及各项 CI 均不受此影响。
- **幻觉率绝对值被系统性高估**。报告中 22.25–22.56% 的 fabrication_ratio 含这批伪幻觉。

处理方式：**不修改冻结评估脚本与产物**（`candidate_scores.json`、`person_caption_metrics.json`
保持原样，页面顶部全局指标直接取自冻结产物未作修正），仅在 demo 显示层用 ⌗ 状态区分两类情况。
若要修正幻觉率绝对值，需改 `normalize_value` 支持列表型 GT 并重跑评估。
