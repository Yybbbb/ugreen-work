# Florence Person Attribute 训练记录

本文档记录基于 reviewed 更新数据重建后的 Florence person attribute 全参训练。五个版本均从 `pretrained/Florence-2-base` 独立初始化，在 GPU 3-6 上串行训练；每轮使用 final checkpoint 在固定测试集推理，再由 `Qwen36-35b-caption` 在 GPU 1-2 上抽取属性并计算指标。

## 数据与评估冻结版本

本次训练使用 reviewed 更新后的 prepared 数据。测试集 caption 为空的样本在准备阶段删除；本批没有空 caption，因此测试集保持 4,328 条。

| 文件 | 样本数 | SHA-256 |
|---|---:|---|
| `data/prepared/train.jsonl` | 30,000 | `4752fe4631a27d873bc053caff16ab712b4a09d23ee6ab1ce98fe14ee7d5c575` |
| `data/prepared/dev.jsonl` | 2,000 | `a725317ca6b3df1a58992975d02b6e9020b6ae9832c22652e23b2f9e3da4b7f9` |
| `data/prepared/test.jsonl` | 4,328 | `4beeb8dfd29917ffa787a0f84c30f3f3be77d82a1e172d23d61c11c7db82772c` |
| `data/prepared/native_replay.jsonl` | 1,000 | `87b6a1c2a1dc9557283402322e83bf1ec7d71628b6043de5d103db51c49a6900` |
| `data/prepared/rl.jsonl` | 5,981 | `d0b42e627e4e8913421915a6c27b3706aaf03bc582c5ff1c4fa1d97354859b60` |
| `data/prepared/test_qwen_attributes.jsonl` | 4,328 | `d95a678714f43c47fad0b8fbbffe84af438cf025d34438acb1a7d5b9d221b0e7` |

GT caption 属性由 Qwen 重新提取，manifest 为 [`test_qwen_attributes.manifest.json`](../data/prepared/test_qwen_attributes.manifest.json)。共提交 4,352 个请求，初始批处理失败 24 条，逐条重试 24 条后最终失败数为 0。

评估固定使用 4,328 条测试样本、greedy Florence 解码、Qwen 字段规范化和 `scripts/evaluate_with_qwen_gt.py`。Qwen-GT 结果位于每个 run 的 `test_inference_*/evaluation_qwen_final/metrics_qwen_gt.json`；每个最终属性文件均为 4,328 行、无缺失 GT、无重复行、无提取失败。

## 版本总览

| 版本 | Run | 数据 | Global batch | 有效 epoch | Optimizer steps | 状态 |
|---|---|---|---:|---:|---:|---|
| V1 | `region_category_person_sft_30k_replay1k_b64` | 30k + 1k replay | 64 | 1 | 485 | 已完成 |
| V2 | `region_category_person_sft_30k_replay1k_b32` | 30k + 1k replay | 32 | 1 | 969 | 已完成 |
| V3 | `region_category_person_sft_30k_replay1k_b32_e1p5` | 30k + 1k replay | 32 | 1.5 | 1,454 | 已完成 |
| V4A | `region_category_person_sft_30k_replay1k_b16_e3_lr125` | 30k + 1k replay | 16 | 3 | 5,814 | 已完成 |
| V4B | `region_category_person_sft_30k_person_only_b16_e3_lr125` | 30k + 0 replay | 16 | 3 | 5,625 | 已完成 |

## 手工 GT 指标

下面的 GT 是测试样本中的人工 `attributes`，预测侧是 Qwen 从 Florence caption 抽取的属性。`Unknown-extra rate` 是 GT 为 unknown/none 的位置仍被抽出非空属性的比例，不等同于经过视觉验证的幻觉率。

| 版本 | Micro P/R/F1 | Soft Micro F1 | Macro-field F1 | Mean field exact | Unknown-extra | 平均词数 | P95 | 18-24 词 | 背景关键词 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 0.6755 / 0.5522 / 0.6076 | 0.6733 | 0.4743 | 0.5522 | 27.38% | 21.75 | 29 | **59.08%** | 0.116% |
| V2 | 0.6836 / 0.5598 / 0.6155 | 0.6796 | 0.4829 | 0.5598 | 26.70% | 21.51 | 29 | 55.91% | 0.139% |
| V3 | 0.6896 / 0.5647 / 0.6209 | 0.6842 | 0.4888 | 0.5647 | 26.01% | 21.42 | 30 | 55.22% | 0.069% |
| V4A replay1k | 0.7056 / 0.5756 / 0.6340 | 0.6947 | 0.5070 | 0.5756 | 24.64% | 21.34 | 30 | 48.59% | 0.069% |
| V4B no-replay | **0.7075 / 0.5769 / 0.6356** | **0.6962** | **0.5147** | **0.5769** | **24.55%** | 21.32 | 30 | 49.12% | 0.069% |

在人工 GT 口径下，V4B 的 Micro F1、Soft Micro F1 和 Macro-field F1 最高；V4A 与 V4B 的差异很小。长训练显著提升属性准确率，但 18-24 词比例从 V1 的 59.08% 降至约 49%，后续 RL 仍应保留长度奖励。

## Qwen-GT 指标（当前主指标）

为消除人工 GT 词汇与 Qwen schema 的错配，下面使用 Qwen 从 GT caption 提取的 `test_qwen_attributes.jsonl` 作为 GT。预测和 GT 经过同一个 `Qwen36-35b-caption` 抽取器，因此这一节是当前版本比较的主指标。

| 版本 | Micro P/R/F1 | Soft Micro F1 | Macro-field F1 | Mean field exact | Unknown-extra | 平均词数 | P95 | 18-24 词 | 背景关键词 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 0.6822 / 0.6118 / 0.6451 | 0.7150 | 0.5098 | 0.6118 | 14.79% | 21.75 | 29 | **59.08%** | 0.116% |
| V2 | 0.6906 / 0.6219 / 0.6545 | 0.7228 | 0.5230 | 0.6219 | 13.84% | 21.51 | 29 | 55.91% | 0.139% |
| V3 | 0.6985 / 0.6292 / 0.6620 | 0.7295 | 0.5324 | 0.6292 | 13.03% | 21.42 | 30 | 55.22% | 0.069% |
| V4A replay1k | 0.7162 / 0.6464 / 0.6795 | 0.7436 | 0.5573 | 0.6464 | 10.99% | 21.34 | 30 | 48.59% | 0.069% |
| V4B no-replay | **0.7183 / 0.6477 / 0.6812** | **0.7447** | **0.5647** | **0.6477** | **10.94%** | 21.32 | 30 | 49.12% | 0.069% |

V1→V2→V3→V4 的属性指标持续提升。相对 V1，V4B 的 Qwen-GT Micro F1 提升 3.61 个百分点，Macro-field F1 提升 5.49 个百分点，unknown-extra rate 降低 3.85 个百分点。V4B 在主指标上略优于 V4A；V4A 的长度合格率略低，二者应在 RL 中共同作为基线参考。

### Qwen-GT 逐字段 positive F1

| 字段 | V1 | V2 | V3 | V4A | V4B |
|---|---:|---:|---:|---:|---:|
| age_group | 0.8356 | 0.8381 | **0.8456** | 0.8427 | 0.8439 |
| gender | 0.8800 | 0.8818 | 0.8842 | 0.8978 | **0.8976** |
| upper_garment.type | 0.4997 | 0.5082 | 0.5166 | 0.5514 | **0.5523** |
| upper_garment.color | 0.5737 | 0.5813 | 0.5857 | 0.6074 | **0.6112** |
| upper_garment.length | 0.7289 | 0.7509 | 0.7623 | 0.7933 | **0.7970** |
| lower_garment.type | 0.7890 | 0.7948 | 0.8060 | **0.8248** | 0.8214 |
| lower_garment.color | 0.5076 | 0.5229 | 0.5461 | 0.5581 | **0.5613** |
| lower_garment.length | 0.7677 | 0.7825 | **0.7927** | 0.7955 | 0.7956 |
| shoes.type | 0.5461 | 0.5558 | 0.5632 | 0.6077 | **0.6051** |
| shoes.color | 0.5426 | 0.5524 | 0.5608 | 0.5821 | **0.5838** |
| head.accessories | 0.3953 | 0.4270 | 0.4334 | 0.4688 | **0.4742** |
| head.hairstyle | 0.2125 | 0.2603 | 0.2770 | 0.3570 | **0.3685** |
| head.hair_color | 0.4767 | 0.4864 | 0.4916 | 0.5150 | **0.5196** |
| head.hair_length | **0.8544** | 0.8538 | 0.8486 | 0.8371 | 0.8351 |
| carried_items.handbag | 0.0577 | 0.0664 | 0.0707 | 0.0871 | **0.0954** |
| carried_items.backpack | 0.2263 | 0.2300 | 0.2488 | 0.3229 | **0.3416** |
| handheld_items.dangerous_item | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0741** |
| handheld_items.mobile_phone | 0.2822 | 0.3216 | 0.3491 | 0.3818 | **0.3867** |

主要增益集中在上衣类型/颜色/长度、下衣类型、鞋、头部配饰、发型、背包和手机。`head.hair_length` 在 V1 最优并随长训练略有回退；危险物正例稀少，V4B 的非零 F1 仍需扩大样本验证。

## 公共训练配置

| 配置 | V1/V2/V3 | V4A/V4B |
|---|---|---|
| 初始化 | `pretrained/Florence-2-base` | `pretrained/Florence-2-base` |
| 训练方式 | 全量参数更新，231,414,016 参数 | 全量参数更新，231,414,016 参数 |
| 任务 | 人物 `<REGION_TO_CATEGORY>` | 人物 `<REGION_TO_CATEGORY>` |
| replay | 1,000 条原生 `<REGION_TO_DESCRIPTION>` | V4A=1,000；V4B=0 |
| GPU | 4 × RTX 6000 Ada（GPU 3-6） | 4 × RTX 6000 Ada（GPU 3-6） |
| 精度 | BF16 AMP，FP32 optimizer state | BF16 AMP，FP32 optimizer state |
| 优化器 | AdamW，betas=(0.9, 0.999)，eps=`1e-8` | AdamW，betas=(0.9, 0.999)，eps=`1e-8` |
| Weight decay | `0.01` | `0.015` |
| Label smoothing | `0.05` | `0.05` |
| 梯度裁剪 | `1.0` | `1.0` |
| Warmup / Scheduler | 5% / cosine，floor=10% | 5% / cosine，floor=3% |
| Vision / Projection / Language LR | `1e-7` / `5e-7` / `1e-6` | `2e-7` / `7.5e-7` / `1.25e-6` |
| Target max length | 64 tokenizer tokens | 64 tokenizer tokens |
| 每 rank train workers | 6，prefetch factor=1 | 6，prefetch factor=1 |
| 每 rank dev workers | 0 | 0 |
| 随机种子 | `20260720` | `20260720` |

## 训练与验证摘要

以下为每轮最后一次完整 dev 评估和训练终点；测试集只评估 final checkpoint。

| 版本 | 最后 dev step | Dev loss | Dev 平均词数 | Dev 18-24 词 | Dev P95 | 训练终点 |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 450 | 1.6270 | 22.23 | 58.59% | 29 | 485 |
| V2 | 900 | 1.5606 | 21.52 | 60.16% | 29 | 969 |
| V3 | 1400 | 1.5048 | 21.66 | 56.25% | 30 | 1,454 |
| V4A | 5800 | 1.3643 | 21.38 | 50.00% | 30 | 5,814 |
| V4B | 5600 | 1.3648 | 21.30 | 47.66% | 30 | 5,625 |

训练日志：

- [V1 train.log](../artifacts/sft/region_category_person_sft_30k_replay1k_b64/train.log)
- [V2 train.log](../artifacts/sft/region_category_person_sft_30k_replay1k_b32/train.log)
- [V3 train.log](../artifacts/sft/region_category_person_sft_30k_replay1k_b32_e1p5/train.log)
- [V4A train.log](../artifacts/sft/region_category_person_sft_30k_replay1k_b16_e3_lr125/train.log)
- [V4B train.log](../artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/train.log)

## 最终模型与评估产物

| 版本 | Final checkpoint | 测试推理 manifest | Qwen-GT 指标 |
|---|---|---|---|
| V1 | [`final`](../artifacts/sft/region_category_person_sft_30k_replay1k_b64/final) | [`manifest.json`](../artifacts/sft/region_category_person_sft_30k_replay1k_b64/test_inference_final_v2/manifest.json) | [`metrics_qwen_gt.json`](../artifacts/sft/region_category_person_sft_30k_replay1k_b64/test_inference_final_v2/evaluation_qwen_final/metrics_qwen_gt.json) |
| V2 | [`final`](../artifacts/sft/region_category_person_sft_30k_replay1k_b32/final) | [`manifest.json`](../artifacts/sft/region_category_person_sft_30k_replay1k_b32/test_inference_final/manifest.json) | [`metrics_qwen_gt.json`](../artifacts/sft/region_category_person_sft_30k_replay1k_b32/test_inference_final/evaluation_qwen_final/metrics_qwen_gt.json) |
| V3 | [`final`](../artifacts/sft/region_category_person_sft_30k_replay1k_b32_e1p5/final) | [`manifest.json`](../artifacts/sft/region_category_person_sft_30k_replay1k_b32_e1p5/test_inference_final/manifest.json) | [`metrics_qwen_gt.json`](../artifacts/sft/region_category_person_sft_30k_replay1k_b32_e1p5/test_inference_final/evaluation_qwen_final/metrics_qwen_gt.json) |
| V4A | [`final`](../artifacts/sft/region_category_person_sft_30k_replay1k_b16_e3_lr125/final) | [`manifest.json`](../artifacts/sft/region_category_person_sft_30k_replay1k_b16_e3_lr125/test_inference_final/manifest.json) | [`metrics_qwen_gt.json`](../artifacts/sft/region_category_person_sft_30k_replay1k_b16_e3_lr125/test_inference_final/evaluation_qwen_final/metrics_qwen_gt.json) |
| V4B | [`final`](../artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final) | [`manifest.json`](../artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/test_inference_final/manifest.json) | [`metrics_qwen_gt.json`](../artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/test_inference_final/evaluation_qwen_final/metrics_qwen_gt.json) |

旧版 run 和旧版 `test_qwen_attributes` 已备份到 `artifacts/sft/reviewed_retrain_backup_20260809_181345/`。

## 对 RL 训练的结论

1. 采用 V4B final 作为当前属性主指标基线：Qwen-GT Micro F1=`0.6812`、Soft Micro F1=`0.7447`、Macro-field F1=`0.5647`。
2. V4A 与 V4B 均明显优于 V1-V3，说明小 batch、较高分层学习率和更长曝光带来稳定属性收益；replay 的边际收益很小，V4B 的主指标略高。
3. 长度是当前主要短板：V1 的 18-24 词比例为 59.08%，V4A/V4B 降到 48.59%/49.12%，平均词数仍在 21.3-21.8，说明是分布变宽而非整体长度漂移。
4. 第一版 RL 使用 `alpha=0.10 → 0.50`：CE 保持自然 caption 锚点，SCST 权重在 748 个更新中逐步增强。长度奖励沿用已确定的分段：18-24 词正奖励，12-17 与 25-28 词轻罚，更远范围重罚。
5. RL 默认从 V4B final 初始化，统一输出到 `artifacts/rl`。每 100 step 使用全量 2,000 条 dev CE 和固定 128 条 greedy reward 选择 top-3，并始终另存不参与淘汰的 final checkpoint。
6. `--resume-from` 必须恢复 optimizer、scheduler、GradScaler、epoch/batch/global-step、top-3 和每个 rank 的 RNG；任何数据、alpha、生成参数或 reward 配置不一致都拒绝继续。
7. RL 评估需同时跟踪 Qwen-GT 属性 F1、18-24 词比例、unknown-extra rate、背景关键词率、空输出率和单句/主体开头约束，不能只优化 caption reward。GT unknown 视为属性不存在；若生成具体值则计为 fabrication。
8. 正式服务固定为 GPU 1 的 `Qwen3.6-27B-FP8`（6097，vLLM 0.19.1，context 65,536，并发 64，batched tokens 8,192，强制 FP8 Marlin）和 GPU 2 的 `Qwen3.5-4B`（6098，context 32,768，并发 128）。两者均为纯文本、默认关闭 thinking。
9. 使用 `python3 scripts/run_rl_reward_services.py status all` 做模型身份和 `OK`/无 reasoning 探针。训练在加载 Florence 前自动执行同一预检并广播结果；`--skip-reward-service-preflight` 只用于诊断且会记录到配置。服务 CLI 不自动替换已有异常容器。

## 评估协议检查清单

每次后续训练至少记录：

- Micro Precision/Recall/F1、Soft Micro F1、Macro-field F1、Mean field exact；
- 18 个固定标量字段 positive F1 与开放 `extra` 的匹配统计；
- unknown-extra rate、Qwen 提取失败数、空输出数、重复行和非法格式数；
- 平均长度、P95、18-24 词比例、单句率、主体开头率、背景关键词率；
- 训练/验证数据 SHA-256、final checkpoint 路径、测试 manifest 和 Qwen-GT manifest。

测试集只用于 checkpoint 冻结后的最终报告，不能反向选择训练超参数。

## RL final 三模型测试评估

冻结对比对象为：

- V4B SFT：`artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final`；
- Qwen RL：`artifacts/rl/region_category_person_scst_reviewed5981_qwen4b/final`；
- lexical RL：`artifacts/rl/region_category_person_scst_reviewed5981_lexical/final`。

报告并排展示 8 组指标：lexical Micro P/R/F1、Qwen4B semantic Micro P/R/F1、
lexical Macro-field F1、fabrication ratio、structure pass rate、平均词数与 18-24
词比例、background leakage rate、invalid output rate。所有差值均相对 V4B，
以 session 为单位做 2,000 次配对 bootstrap。主排序分数为两种 Micro F1 的
等权平均，但必须先通过属性、fabrication、句式、长度、背景和格式门槛；小于
等于 0.5 个百分点或 CI 包含 0 的提升不视为明确收益。

执行入口是 `scripts/run_person_attribute_rl_evaluation.sh`。runner 等待 lexical
RL final 完成并释放 GPU 3-6 后，依次完成两个 RL checkpoint 的 test 推理，再用
6097 的 `Qwen3.6-27B-FP8` 统一抽取三组 caption，并用 6098 的
`Qwen3.5-4B` 统一裁判。断电后 Florence rank 输出、抽取 journal 和 judge cache
均可继续。实时日志及最终 JSON/Markdown 报告位于 `artifacts/rl/evaluation/`。
