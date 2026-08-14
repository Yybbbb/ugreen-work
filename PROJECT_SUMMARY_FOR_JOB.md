# segment-color 项目梳理与求职版总结

> 求职定位一句话：本项目围绕“行人/人体区域分割 + 上下装颜色识别 + 低置信拒答”构建了一套从数据采集、自动标注、清洗、训练、评估到部署策略分析的完整视觉属性识别 pipeline，最终在 RGB-only PP-LiteSeg-STDC2 多任务模型上取得 `mIoU 0.6607`、上装颜色 `macro-F1 0.7730`、下装颜色 `macro-F1 0.6772` 的验证集结果。

本文档用于后续简历、面试复盘和项目交接。重点覆盖 `data`、`extradata`、`experiment`、`statistics`、`refuse-answer`，并补充 `labeling`、`scripts` 中和数据闭环相关的部分。

## 1. 项目概览

### 1.1 项目目标

项目目标是对人像 crop 或行人区域进行细粒度视觉属性理解：

1. 输出人体属性分割 mask，覆盖背景、上装、下装、连衣裙、帽子、眼镜、口罩、头发等 8 类区域。
2. 对上装和下装分别预测颜色类别。
3. 对小图、颜色复杂、多色/条纹/格纹等不可靠场景建立拒答或降置信策略。
4. 构建可扩展的数据生产流程，包括视频抽帧、行人检测裁剪、SAM3 分割标注、Qwen-VL 颜色标注、人工审核、去重清洗、难例/稀有色补充。

### 1.2 核心产出

| 模块 | 产出 |
| --- | --- |
| 数据 | 8.9 万级主数据集、2.6 万级稀有色数据、1.4 万级真实场景裁剪数据，包含图像、分割标注、颜色标签和部分主体/携带物标签 |
| 模型 | PaddleSeg/PP-LiteSeg-STDC2 多任务模型，同时做 8 类分割、上装颜色分类、下装颜色分类 |
| 实验 | 对比 pose prior、zero-pose、RGB-only、分阶段训练、从零联合训练等方案 |
| 评估 | 全局分割指标、颜色分类指标、按类别 IoU、按像素占比的检出率、badcase HTML 可视化 |
| 改进 | 数据清洗、稀有色补齐、RGB-only 方案、unknown 类处理、OHEM、颜色扰动隔离、低分辨率拒答、多色拒答 |

### 1.3 推荐简历表述

可以在简历中压缩成 3 到 5 条：

- 负责构建人体属性分割与服装颜色识别数据闭环，整合 LIP、IPC 分割标注、UPAR 稀有色样本和真实场景检测裁剪数据，形成 8.9 万级主数据、2.6 万级稀有色补充数据和 1.4 万级真实场景样本。
- 基于 PaddleSeg 改造 PP-LiteSeg-STDC2 为多任务网络，主干共享，分割头输出 8 类人体属性 mask，双颜色 head 分别预测上装/下装 12 类颜色。
- 设计并对比 pose prior、zero-pose、RGB-only、分阶段训练、从零联合训练等方案，最终 RGB-only 方案在验证集达到 `mIoU 0.6607`、上装 `macro-F1 0.7730`、下装 `macro-F1 0.6772`，综合分数 `0.6929`。
- 建立按像素占比、类别、badcase 的评估分析流程，发现帽子、眼镜、口罩等小目标属性受面积影响明显，并输出 HTML 可视化报告辅助迭代。
- 针对部署风险设计拒答策略：小 crop 阈值规则、多色/条纹/格纹纯色门控，降低低分辨率和颜色不纯场景下的错误颜色输出。

## 2. 目录结构总览

```text
segment-color/
├── README.md                         # 项目原始说明
├── data/                             # 训练/验证数据、清洗结果、SAM3 标注结果、split 文件
├── extradata/                        # 额外视频抽帧、UPAR 数据处理、补充标注脚本
├── experiment/PaddleSeg/             # PaddleSeg 训练代码、模型、配置、日志、实验报告
├── labeling/                         # SAM3/Qwen-VL 自动标注、人工审核、标注质量分析
├── scripts/                          # 行人检测裁剪、SAM3 批量标注等数据生产脚本
├── statistics/                       # 模型预测统计、像素占比 vs 检出率、badcase 可视化
├── refuse-answer/                    # 小图拒答、多色拒答策略
├── logs/                             # 部分运行日志
├── color_label_distribution_train.md # 颜色标签分布统计
└── yolo11n-pose.pt                   # YOLO pose/crop 相关权重
```

各目录在项目链路中的位置：

```text
原始视频/图片
  -> extradata/scripts 抽帧
  -> scripts YOLO 行人检测裁剪
  -> labeling / scripts SAM3 分割标注
  -> labeling / extradata Qwen-VL 颜色标注
  -> data 清洗、去重、合并、split
  -> experiment/PaddleSeg 训练多任务模型
  -> statistics 评估与 badcase
  -> refuse-answer 部署拒答策略
```

## 3. 数据部分：data

`data` 是项目的核心数据目录，包含主训练集、稀有色补充集、真实场景补充集、清洗列表和 split 文件。

### 3.1 标签体系

#### 分割类别

模型输出 8 类分割：

| id | 类别 | 含义 |
| --- | --- | --- |
| 0 | background | 背景 |
| 1 | upper | 上装 |
| 2 | lower | 下装 |
| 3 | dress | 连衣裙/裙装整体 |
| 4 | hat | 帽子 |
| 5 | glasses | 眼镜 |
| 6 | mask | 口罩 |
| 7 | hair | 头发 |

#### 颜色类别

上装和下装颜色分类使用 12 类：

```text
unknown, black, white, gray, red, yellow, green, blue, purple, pink, orange, brown
```

其中 `unknown` 的处理在不同实验中不同：

- 早期 CE loss 中可通过 `ignore_index` 忽略 unknown，避免噪声影响。
- RGB-only 最优实验里使用 `ce_ignore_index=-1`，即把 unknown 也作为一个有效类别训练，这提升了实际闭集/开集边界表达。

### 3.2 主数据集：LIP_clothes_accessory_unified

主数据集路径：

```text
data/LIP_clothes_accessory_unified/
```

规模：

| 子目录/文件 | 数量 |
| --- | ---: |
| images | 89,221 |
| annotations | 89,221 |
| color-label | 53,269 |

来源构成：

| 来源 | 样本数 |
| --- | ---: |
| ipc_seg_annotation | 34,010 |
| lip | 40,462 |
| upper_lower_person_export | 14,749 |
| 合计 | 89,221 |

每个来源的属性 mask 数量：

| 来源 | upper | lower | dress | hat | glasses | mask | hair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ipc_seg_annotation | 29,765 | 22,590 | 15,841 | 14,294 | 15,425 | 1,363 | 24,531 |
| lip | 36,654 | 25,402 | 1,642 | 10,421 | 2,347 | 626 | 28,072 |
| upper_lower_person_export | 14,749 | 14,749 | 0 | 1,600 | 4,676 | 1,060 | 12,873 |

数据特点：

- 覆盖上装、下装、连衣裙、帽子、眼镜、口罩、头发等多个属性区域。
- LIP 提供人体解析基础，IPC/导出数据补充真实业务或内部采集分布。
- 色彩标签覆盖部分数据，分割标签覆盖更大规模数据，因此训练时需要处理颜色标签缺失。
- 图片存储使用 hardlink 方式，降低重复拷贝成本。

### 3.3 训练/验证 split

项目中存在多个 split，口径不同，面试时需要讲清楚。

| split | train | val | 用途 |
| --- | ---: | ---: | --- |
| `data/splits/train.txt` / `val.txt` | 71,345 | 7,927 | RGB-only 最优实验使用的数据口径 |
| `data/splits-new/train.txt` / `val.txt` | 84,295 | 9,366 | 扩展后的新 split，覆盖更多数据 |
| `data/LIP_clothes_accessory_unified/splits/clothes_accessory_poseprior/train.txt` / `val.txt` | 46,878 | 6,391 | pose prior 系列实验使用的数据口径 |

注意点：

- `pose prior` 系列和 `RGB-only` 系列不是完全相同的数据量，因此横向比较时要说明数据口径差异。
- 早期 README 中的 `mIoU=0.636`、上装颜色 `Acc=0.834`、下装颜色 `Acc=0.816` 属于早期 pose prior 结果，不是最终 RGB-only 最优结果。

### 3.4 颜色标签分布

`color_label_distribution_train.md` 统计了 `data/splits/train.txt` 的颜色分布，训练集共 71,345 条。

#### 上装主色分布

| 颜色 | 数量 | 占比 |
| --- | ---: | ---: |
| black | 14,163 | 19.89% |
| white | 13,428 | 18.86% |
| gray | 8,135 | 11.42% |
| red | 6,564 | 9.22% |
| blue | 5,752 | 8.08% |
| unknown | 5,232 | 7.35% |
| brown | 4,659 | 6.54% |
| green | 4,302 | 6.04% |
| pink | 2,965 | 4.16% |
| yellow | 2,944 | 4.13% |
| purple | 2,360 | 3.31% |
| orange | 711 | 1.00% |

#### 下装主色分布

| 颜色 | 数量 | 占比 |
| --- | ---: | ---: |
| black | 29,295 | 41.10% |
| blue | 14,889 | 20.89% |
| gray | 7,478 | 10.49% |
| brown | 6,301 | 8.84% |
| white | 5,610 | 7.87% |
| unknown | 2,077 | 2.91% |
| pink | 1,530 | 2.15% |
| red | 1,506 | 2.11% |
| green | 1,461 | 2.05% |
| yellow | 596 | 0.84% |
| purple | 362 | 0.51% |
| orange | 180 | 0.25% |

关键结论：

- 下装颜色严重长尾，黑色和蓝色合计超过 60%，橙色、紫色、黄色极少。
- 上装颜色相对均衡，但橙色、紫色、黄色仍偏少。
- 多色样本比例：上装约 10.26%，下装约 1.98%。
- 因为最终任务是颜色分类，单看 overall accuracy 容易被黑/白/蓝大类支配，所以需要重点看 macro-F1、macro-precision、macro-recall。

### 3.5 稀有色补充：UPAR_rare_color

路径：

```text
data/UPAR_rare_color/
```

规模：

| 子目录/文件 | 数量 |
| --- | ---: |
| images | 26,699 |
| annotations | 26,699 |
| color-label | 26,003 |

来源：

| 来源 | 数量 |
| --- | ---: |
| PA100k | 19,288 |
| Market1501 | 5,167 |
| PETA | 2,244 |

相关文件：

```text
extradata/upar-dataset/rare_color_samples.txt      # 28,384
data/UPAR_rare_color/rare_color_single_color.txt   # 26,003
```

用途：

- 针对黄色、紫色、橙色、粉色、绿色等长尾颜色补数据。
- 用 SAM3 自动生成分割标注。
- 用颜色标签文件补足上/下装颜色分类监督。

SAM3 标注统计：

| 指标 | 值 |
| --- | ---: |
| total | 26,701 |
| success | 26,699 |
| errors | 2 |
| elapsed | 12,107.03 秒 |
| rate | 2.205 img/s |
| server | `localhost:8010` |
| conf threshold | 0.3 |

### 3.6 真实场景裁剪：object_detection_0309-0429

路径：

```text
data/object_detection_0309-0429/
```

规模：

| 子目录 | 数量 |
| --- | ---: |
| images | 14,389 |
| annotations | 14,389 |
| color-label | 14,389 |
| subject-label | 2,694 |

场景分布：

| 场景 | 样本数 |
| --- | ---: |
| 2026_04_03_cat_cafe | 4,682 |
| 2026_4_9_ikea | 2,629 |
| 2026_3_20_expo | 2,079 |
| 2026_3_12_euro_city | 1,250 |
| 2026_3_17_hardware_store | 672 |
| 2026_3_09_park | 559 |
| 2026_4_8_cat_cafe | 543 |
| 2026_4_16_transport_hub | 444 |
| 2026_3_25_urban_village | 424 |
| 2026_3_26_cat_cafe | 337 |
| 2026_4_28_park | 276 |
| 2026_4_21 | 151 |
| 2026_3_30_mall | 148 |
| 2026_4_17_residential_area | 144 |
| 2026_4_29_park | 51 |

用途：

- 补足真实业务场景中的视角、遮挡、光照、低清、非标准姿态。
- 验证从公开数据到真实场景的迁移能力。
- 给后续小图拒答、多色拒答、主体/携带物属性扩展提供真实样本。

SAM3 标注统计：

| 指标 | 值 |
| --- | ---: |
| total_seen | 14,389 |
| success | 14,386 |
| skipped | 3 |
| errors | 0 |
| elapsed | 5,938.49 秒 |
| rate | 2.423 img/s |
| server | `localhost:8017` |
| GPU | 7 |

### 3.7 数据清洗与去重

关键文件：

```text
data/all-data-clean-image.txt
data/pose_yolo26n/lists/
```

清洗结果：

| 文件/列表 | 数量 |
| --- | ---: |
| `all-data-clean-image.txt` | 52,090 |
| `pose_yolo26n/lists/database` | 9,864 |
| `pose_yolo26n/lists/drop` | 34,056 |
| `pose_yolo26n/lists/no_person` | 1,896 |
| `pose_yolo26n/lists/reid` | 43,405 |
| `pose_yolo26n/lists/usable_database_or_reid` | 53,269 |

清洗策略：

- 使用 dHash 感知哈希做近重复检测。
- Hamming distance 阈值为 8。
- 按文件名前缀或 Objects365 相关字段分组，降低全量两两比较复杂度。
- 使用 greedy independent set 保留代表样本。
- 哈希缓存保存在 `.hash_cache.pkl`，避免重复计算。

这部分适合面试展开：

- 为什么不能只按文件名去重：视频抽帧、同一行人连续帧、不同来源重复 crop 都可能产生近似图。
- 为什么用 dHash：速度快，对轻微缩放、压缩、亮度变化相对鲁棒。
- 为什么要按组比较：全量 O(N^2) 不可接受，先按视频/来源/前缀分桶能明显降低计算量。

## 4. 额外数据生产：extradata

`extradata` 主要负责从额外视频和公开属性数据中生产候选样本。

### 4.1 视频抽帧

关键脚本：

```text
extradata/scripts/extract_frames_batch.py
```

策略：

- 输入来自 `.dat` 视频文件目录。
- 每 2 秒抽 1 帧，即 `FPS="1/2"`。
- 输出分辨率使用 `scale=1920:-1`。
- `workers=2`，避免 ffmpeg 并发过高导致 OOM。
- 跳过小于 10MB 的异常文件。
- 对已经抽过帧的视频做跳过，支持断点续跑。

这部分体现的是工程稳定性：

- 大批量视频处理时，最重要的不是单次命令能跑，而是可恢复、可跳过、可控并发。
- 抽帧间隔选择 2 秒，是在数据多样性和重复帧冗余之间的折中。

### 4.2 UPAR 稀有色挖掘

相关路径：

```text
extradata/upar-dataset/
extradata/upar-dataset/rare_color_samples.txt
```

作用：

- 从 UPAR 相关数据源中筛出稀有颜色样本。
- 缓解训练集颜色长尾，尤其是下装黄色、紫色、橙色、粉色等类别样本过少的问题。
- 结合 SAM3 分割标注，形成可直接参与训练或评估的数据。

### 4.3 Qwen 颜色标注与清洗

相关脚本：

```text
extradata/scripts/qwen_color_annotate.py
extradata/scripts/qwen_color_clean.py
```

主要作用：

- 使用 Qwen-VL 对上装/下装颜色进行自动标注。
- 对模型输出做结构化解析和清洗。
- 过滤不合规、无法解析或明显冲突的输出。

自动颜色标注需要重点解决：

- 输出格式不稳定。
- 多色描述和单色分类之间的映射。
- `unknown`、遮挡、无下装、连衣裙等边界情况。
- 模型主观色彩判断和训练标签体系之间的一致性。

## 5. 标注系统：labeling

`labeling` 目录连接了自动标注和人工审核，是数据质量闭环的关键。

### 5.1 SAM3 分割标注

关键脚本：

```text
labeling/annotate_ipc.py
```

能力：

- 调用 SAM3 HTTP 服务做人体属性区域标注。
- 支持多 GPU、多端口并发服务。
- 每张图可发起多 prompt 标注。
- 支持 timeout、retry、断点续跑。
- 按 confidence threshold 过滤低置信 mask。

这一部分可以在面试中强调：

- 标注不是一次性脚本，而是一个可恢复的批处理系统。
- 大规模自动标注必须考虑服务稳定性、失败重试、增量处理和输出校验。

### 5.2 Qwen-VL 颜色标注

关键文件：

```text
labeling/qwen_upper_lower_color.py
labeling/qwen_upper_lower_color_prompt.txt
```

目标：

- 让视觉语言模型识别上装和下装颜色。
- 约束输出到固定颜色集合。
- 将自然语言输出转换为结构化 JSON。

标注难点：

- 光照、阴影、反光会影响颜色判断。
- 多色、花纹、条纹需要决定主色或拒答。
- 上装/下装区域可能被包、手、桌子、遮挡物影响。
- 连衣裙样本需要处理 upper/lower/dress 的语义边界。

### 5.3 人工审核与质量分析

相关文件：

```text
labeling/server.py
labeling/export_review_samples.py
labeling/generate_annotation_review.py
labeling/review_samples.csv
labeling/review_samples.json
labeling/annotation_review.html
```

作用：

- 抽样导出标注结果。
- 生成 HTML 审核页面。
- 支持人工检查 SAM3 mask 和 Qwen 颜色标签。
- 为后续修正规则、重新标注和 badcase 分析提供依据。

## 6. 数据生产脚本：scripts

`scripts` 更偏向从真实视频帧中生成训练样本。

关键脚本：

```text
scripts/crop_persons_from_frames.py
scripts/person_crop_pipeline.py
scripts/sam3_annotate_upar_rare_color.py
scripts/sam3_annotate_object_detection_0309_0429.py
```

主要流程：

1. 使用 YOLO11n-pose 检测图像中的人。
2. 按检测框裁剪 person crop。
3. 保存 crop、pose/keypoint、来源信息。
4. 对 crop 调用 SAM3 生成分割 mask。
5. 调用 Qwen 或其他规则生成颜色标签。
6. 汇总为统一数据格式。

工程价值：

- 把“模型训练”前面的数据生产流程工具化。
- 支持真实场景持续增量进入训练集。
- 为后续主动学习、难例挖掘、线上数据回流打基础。

## 7. 模型与实验：experiment/PaddleSeg

实验目录：

```text
experiment/PaddleSeg/
```

重点子目录：

```text
experiment/PaddleSeg/experiments/pose_prior_stdc2/
experiment/PaddleSeg/experiments/clothes_accessory_poseprior_stdc2/
experiment/PaddleSeg/experiments/clothes_color_poseprior_stdc2/
experiment/PaddleSeg/experiments/clothes_color_rgb_only/
experiment/PaddleSeg/experiments/clothes_color_zeropose_ablation/
experiment/PaddleSeg/experiments/pidnet_s_4cls/
experiment/PaddleSeg/experiments/pp_liteseg_stdc2_*          # 训练输出、checkpoint、log
experiment/PaddleSeg/experiments/smoke_*                     # 小规模冒烟实验
experiment/PaddleSeg/experiments/upperlower_web_demo/         # demo/可视化入口
```

### 7.0 experiments 目录怎么读

`experiment/PaddleSeg/experiments` 里混合了两类内容：

1. 实验代码目录  
   例如 `clothes_color_poseprior_stdc2/`、`clothes_color_rgb_only/`、`clothes_color_zeropose_ablation/`。这些目录里有 dataset、model、loss、train script、config、分析文档。

2. 训练输出目录  
   例如 `pp_liteseg_stdc2_clothes_color_2head_rgb_3ch_256_optC_scratch/`。这些目录里主要是 `checkpoints/`、`train.log`、`launch_log/`、结果对比报告。

因此看实验时应该按“代码目录 -> 配置 -> 输出目录 -> 结果报告”的顺序看，而不是直接从 checkpoint 目录开始。

#### 7.0.1 实验演进主线

| 阶段 | 代表目录 | 目标 | 结论/作用 |
| --- | --- | --- | --- |
| 4 类 upper/lower 分割基线 | `pp_liteseg_stdc2_upperlower_4cls_256_20k/`、`pp_mobileseg_tiny_upperlower_4cls_256_20k/`、`pidnet_s_4cls/` | 先验证上装/下装基础分割能力 | 建立早期分割 baseline，也比较不同轻量模型 |
| 5 通道 pose prior | `pose_prior_stdc2/` | 给 upper/lower/dress 引入上半身和腿部先验 | 验证 pose heatmap 能否帮助结构定位 |
| 8 类 clothes + accessory | `clothes_accessory_poseprior_stdc2/` | 从 4 类扩展到 8 类人体属性分割 | 生成后续颜色多任务的 segmentation checkpoint |
| 6 通道颜色多任务 | `clothes_color_poseprior_stdc2/` | 在 8 类分割上增加 upper/lower 双颜色 head | 系统比较 Option A/B/C 三种训练策略 |
| zero-pose 消融 | `clothes_color_zeropose_ablation/` | 把 3 个 pose 通道置零，隔离 pose 贡献 | 证明 pose prior 对分割帮助有限，对 upper/稀有色有一定帮助 |
| RGB-only 最终方案 | `clothes_color_rgb_only/` | 去掉 pose，使用 3 通道 RGB 和更大数据 | 最优方案，指标和部署复杂度都更好 |
| smoke test | `smoke_*` | 快速验证训练脚本、配置、数据读取能跑通 | 降低长训练前的配置风险 |

这条主线是面试时最容易讲清楚的版本：

```text
先做基础分割 -> 加 pose prior -> 扩到 8 类属性 -> 加颜色多任务 ->
做训练策略对比 -> 做 zero-pose 消融 -> 回到更简单的 RGB-only 并扩大数据
```

#### 7.0.2 代码目录与输出目录对应关系

| 实验代码目录 | 主要输出目录 | 说明 |
| --- | --- | --- |
| `pose_prior_stdc2/` | `pp_liteseg_stdc2_upperlower_4cls_poseprior_5ch_256_20k/` | 4 类 upper/lower + 5ch pose prior |
| `clothes_accessory_poseprior_stdc2/` | `pp_liteseg_stdc2_clothes_accessory_8cls_poseprior_6ch_256_40k/` | 8 类 accessory 分割，给 Option A/Phase1 提供分割预训练 |
| `clothes_color_poseprior_stdc2/` | `pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256_optA_from_seg/` | Option A：加载 8 类分割 ckpt，颜色头随机 |
| `clothes_color_poseprior_stdc2/` | `pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256_phase1_6k_4gpu_v2/` | Phase1：冻结 backbone/seg，只训练颜色头 |
| `clothes_color_poseprior_stdc2/` | `pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256_optB_from_phase1/` | Option B：从 Phase1 resume，全网络微调 |
| `clothes_color_poseprior_stdc2/` | `pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256_optC_scratch/` | Option C：从 ImageNet zero-pad backbone 开始联合训练 |
| `clothes_color_zeropose_ablation/` | `pp_liteseg_stdc2_clothes_color_2head_zeropose_6ch_256_optA/B/C_*` | pose 通道置零的消融实验 |
| `clothes_color_rgb_only/` | `pp_liteseg_stdc2_clothes_color_2head_rgb_3ch_256_optC_scratch/` | 最终 RGB-only 最优方案 |

#### 7.0.3 关键代码文件职责

| 文件 | 职责 | 面试可讲点 |
| --- | --- | --- |
| `clothes_color_poseprior_stdc2/clothes_color_model.py` | 定义 `PPLiteSegWithColorHeads` 和两个颜色 head | 如何在分割 decoder 特征上挂轻量分类头 |
| `clothes_color_poseprior_stdc2/clothes_color_poseprior_dataset.py` | 读取 RGB、pose、seg mask、颜色 label | 多模态输入、label cache、颜色 JSON 容错解析 |
| `clothes_color_poseprior_stdc2/losses.py` | OHEM 分割 loss + 双颜色 CE/BCE + joint loss | 多任务 loss 权重设计和 unknown 处理 |
| `clothes_color_poseprior_stdc2/train_color_poseprior.py` | 训练和验证主循环 | 同时统计分割指标和颜色指标 |
| `clothes_color_poseprior_stdc2/run_three_plans.sh` | A/B/C 三方案启动脚本 | 实验控制变量和可复现性 |
| `clothes_color_rgb_only/dataset_rgb.py` | 3 通道 RGB-only dataset | 规避 BGR/RGB 二次转换，复用统一 mask/label 解析 |
| `clothes_color_rgb_only/configs/optC_scratch_rgb.yml` | RGB-only 最优配置 | 更大数据、更高 LR、warmup、上下装独立 loss 权重 |
| `clothes_color_rgb_only/val_comparison.md` | 三方案最终对比 | 最终结论来源 |
| `clothes_color_zeropose_ablation/dataset_zeropose.py` | pose prior 强制置零 | 严格消融 pose 通道贡献 |
| `clothes_color_zeropose_ablation/analysis.md` | zero-pose 分析报告 | 支撑“pose 是锦上添花，不是核心” |

#### 7.0.4 配置设计重点

实验配置主要保存在各代码目录的 `configs/` 下。最关键的是：

```text
clothes_color_poseprior_stdc2/configs/pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256x256_optA_from_seg.yml
clothes_color_poseprior_stdc2/configs/pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256x256_optB_from_phase1.yml
clothes_color_poseprior_stdc2/configs/pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256x256_phase2_scratch.yml
clothes_color_rgb_only/configs/optC_scratch_rgb.yml
clothes_color_zeropose_ablation/configs/optA_from_seg_zeropose.yml
clothes_color_zeropose_ablation/configs/optB_from_phase1_zeropose.yml
clothes_color_zeropose_ablation/configs/optC_scratch_zeropose.yml
```

几个配置层面的关键点：

- `freeze_mode=color_heads_only` 用于 Phase1，只训练颜色 head，保护已有分割特征。
- `seg_pretrained` 用于只加载分割模型权重，不恢复 optimizer。
- `--resume_model` 用于 Option B，恢复完整模型、optimizer、LR scheduler。
- 6 通道 pose prior 的第一层卷积来自 3 通道 STDC2 权重 zero-pad：RGB 通道继承 ImageNet，pose 通道初始为 0。
- RGB-only 使用官方 3 通道 STDC2 预训练权重，链路更标准。
- RGB-only 中 `ce_ignore_index=-1`，即 unknown 不再被忽略，而是作为 12 类之一纳入训练和评估。
- RGB-only 中颜色 loss 从统一 `color_weight` 演进为 `upper_color_weight=0.6`、`lower_color_weight=0.4`，说明上下装标签噪声和类别长尾被分开处理。

#### 7.0.5 launch_log、checkpoint、train.log 怎么看

训练输出目录通常包含：

```text
checkpoints/
launch_log/
train.log 或 train_nohup.log
```

含义：

- `checkpoints/best_model/`：当前实验保存的最佳模型。
- `checkpoints/iter_xxx/`：固定间隔保存的中间 checkpoint，便于回溯过拟合或早停点。
- `checkpoints/checkpoint_list.txt`：PaddleSeg 保存 checkpoint 的列表。
- `launch_log/backup_env.*.json`：分布式训练环境备份。
- `launch_log/workerlog.*`：多卡 worker 日志。
- `train.log` / `train_nohup.log`：训练 loss、验证指标、best score 的主要来源。

这也是为什么项目里同时保留 `val_comparison.md`、`THREE_PLANS_ANALYSIS.md`、`color_metrics_report.md`：长日志不适合直接交接，需要整理成实验报告。

### 7.1 模型结构

核心模型：

```text
PPLiteSegWithColorHeads
```

结构：

```text
输入图像
  -> STDC2 backbone
  -> PPLiteSeg decoder/head
  -> segmentation logits: 8 类人体属性分割
  -> upper color head: 12 类上装颜色
  -> lower color head: 12 类下装颜色
```

颜色 head 结构：

```text
Conv2D(1x1)
BatchNorm
ReLU
AdaptiveAvgPool2D(1)
Linear(32 -> 64)
ReLU
Linear(64 -> 12)
```

设计动机：

- 分割任务提供人体局部区域语义。
- 颜色分类复用 decoder 高分辨率特征，不额外引入复杂 backbone。
- 上装和下装颜色独立 head，避免一个分类器混淆上下半身区域。

### 7.2 输入方案

项目尝试了三种核心输入方案。

#### 方案一：RGB + pose prior，6 通道

输入：

```text
RGB 3 通道 + pose prior 3 通道
```

pose prior 由关键点生成：

- head 区域
- upper body 区域
- leg 区域

特点：

- 通过人体关键点给模型提供结构先验。
- 对上装/下装区域定位可能有帮助。
- 但增加了数据预处理复杂度，也可能引入关键点检测错误。

#### 方案二：zero-pose ablation，6 通道但 pose 全 0

目的：

- 验证 pose prior 本身是否真的带来收益。
- 控制输入维度为 6 通道，排除仅因网络结构变化导致的差异。

结论：

- zero-pose 和 pose prior 的分割 mIoU 非常接近。
- pose prior 对上装颜色和部分稀有颜色有一定帮助，但不是主要收益来源。
- 数据规模和颜色标签质量比 pose prior 更关键。

#### 方案三：RGB-only，3 通道

最终最佳方案：

- 输入只使用 RGB。
- 不依赖 pose 预处理。
- 使用更大训练 split。
- 工程部署更简单，推理链路更短。

结果表明：

- RGB-only 在最终验证集中超过 pose prior 方案。
- 说明在这个项目中，“更多更干净的数据 + 合理训练策略”比额外 pose 通道更有效。

### 7.3 Dataset 设计

#### pose prior dataset

关键点：

- 读取 RGB 图像、分割 annotation、颜色 label。
- 基于 keypoints 生成 3 通道 pose prior。
- 支持 pose cache 和 label cache，减少重复计算。
- 颜色 label 支持 CE 单标签和 BCE multi-hot 两种形式。
- 对 RGB 做颜色增强时，保持 pose channel 不被颜色扰动污染。

这点很重要：

> 如果对 6 通道整体做 RandomDistort，会把 pose prior 当成颜色通道一起扰动，造成输入语义错误。因此实现中需要只对 RGB 做颜色增强，pose channel 保持原值。

#### RGB-only dataset

关键点：

- 输入 3 通道 RGB。
- split 文件包含 image、annotation、color_label 三列。
- 使用 OpenCV 读取后显式转 RGB，并设置 PaddleSeg transform 的 `to_rgb=False`，避免 BGR/RGB 二次转换。
- 使用 label cache 加速标签解析。

这个细节适合面试提：

> 视觉训练中 BGR/RGB 通道顺序错误很隐蔽，模型可能能收敛但效果异常。项目中通过统一读取和 transform 配置避免了重复通道转换。

### 7.4 Loss 设计

关键 loss：

```text
OhemCELoss
DualHeadCELoss
DualHeadBCELoss
SegColorJointLoss
```

联合 loss：

```text
total_loss =
  seg_weight * segmentation_loss
  + upper_color_weight * upper_color_loss
  + lower_color_weight * lower_color_loss
```

RGB-only 最优配置中：

```yaml
seg_weight: 1.0
upper_color_weight: 0.6
lower_color_weight: 0.4
```

OHEM 设置：

```text
thresh = 0.7
min_kept = 131072
ignore_index = 255
```

作用：

- 分割中背景和大区域类别占比高，小目标类别如 glasses、mask 像素少。
- OHEM 强制关注困难像素，缓解小目标属性被大类淹没的问题。
- 上下装颜色 loss 分开加权，避免下装长尾噪声过度影响整体训练。

### 7.5 训练方案对比

`clothes_color_poseprior_stdc2/THREE_PLANS_ANALYSIS.md` 对 pose prior 6 通道方案做了三种训练策略对比。

| 方案 | 初始化/策略 | 训练重点 |
| --- | --- | --- |
| Option A | 加载 8 类分割预训练，颜色 head 随机初始化 | 保住分割能力，再学颜色 |
| Option B | 先冻结 backbone 训练颜色 head，再整体 resume | 分阶段迁移 |
| Option C | 从零联合训练 | 分割和颜色一起适配 |

pose prior split：46,878 train / 6,391 val。

结果：

| 指标 | Option A | Option B | Option C |
| --- | ---: | ---: | ---: |
| combined score | 0.5577 | 0.5858 | 0.6375 |
| seg mIoU | 0.6638 | 0.6412 | 0.6402 |
| upper Acc | 0.7763 | 0.8111 | 0.8351 |
| upper macro-F1 | 0.5582 | 0.6392 | 0.7054 |
| lower Acc | 0.7717 | 0.8115 | 0.8241 |
| lower macro-F1 | 0.3452 | 0.4093 | 0.5496 |
| color mean F1 | 0.4517 | 0.5243 | 0.6275 |

关键结论：

- Option A 分割 mIoU 最高，但颜色 macro-F1 明显落后。
- Option C 综合最好，说明颜色任务不是简单在分割模型上加线性头就能解决，需要 backbone/decoder 共同适配。
- 下装颜色更难，尤其长尾类别更明显。

#### 7.5.1 A/B/C 三方案的真实差异

三方案模型结构完全一样，差异不在网络，而在训练起点和优化路径。

| 维度 | Option A | Option B | Option C |
| --- | --- | --- | --- |
| backbone 起点 | 8 类分割训练后的 backbone | Phase1 checkpoint | ImageNet 3ch -> 6ch zero-pad |
| pose 通道第一层权重 | 已经在分割任务中学过 pose 结构 | Phase1 中基本仍接近 0 | 初始为 0 |
| segmentation head | 来自 8 类分割 best | 来自 Phase1 | 随机或随主训练学习 |
| color head | 随机 | Phase1 已预热 | 随机 |
| optimizer 状态 | 不恢复 | `--resume_model` 恢复 | 不恢复 |
| 训练目标倾向 | 保护分割 | 颜色微调 | 多任务共同适配 |

这个对比说明一个重要事实：

> “加载分割预训练”不等于“颜色任务一定更好”。分割预训练会让模型更稳定地保留 mask 能力，但颜色分类需要区域级外观特征，过强的分割初始化可能反而让颜色 head 适配不足。

#### 7.5.2 Phase1 的意义和局限

Phase1 的设计：

```text
freeze backbone + ppseg_head + seg_heads
只训练 upper_color_head / lower_color_head
训练 6,000 iter
LR = 0.01
```

目标：

- 在稳定分割特征上快速训练颜色头。
- 避免刚开始颜色 head 随机时扰动 backbone。
- 给 Option B 一个更好的颜色分类起点。

实验观察：

- Option B 的颜色 loss 起点明显低于 Option A。
- Option B 的综合分数高于 Option A。
- 但 Option B 最终仍低于 Option C，说明只预热颜色 head 不能完全解决多任务特征适配问题。

面试时可以这样总结：

> Phase1 是一个合理的工程折中，可以更快得到可用颜色 head，但最终最佳方案说明颜色任务需要 backbone 和 decoder 一起重新适配，而不是只训练最后的小头。

#### 7.5.3 Loss 曲线观察

从 `THREE_PLANS_ANALYSIS.md` 看：

- Option A 的 `loss_seg` 基本不动，说明分割模型已经收敛，训练主要在补颜色 head。
- Option B 因为从 Phase1 resume，颜色 loss 起点更低，但解冻后分割 loss 略升。
- Option C 从零开始时 `loss_seg` 高，但很快下降到和其他方案接近；颜色 loss 下降最快，最终接近 0。

这支撑了一个判断：

> 多任务训练不是“分割模型 + 附加分类器”的简单拼接，而是共享特征如何同时服务 dense prediction 和 global/region color classification 的问题。

#### 7.5.4 PosePrior OptC 的 checkpoint 选择

`pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256_optC_scratch/color_metrics_report.md` 对 Option C 做了更细的颜色指标分析。

关键现象：

| 指标 | 颜色最佳点 iter 11000 | final iter 40000 |
| --- | ---: | ---: |
| upper macro-F1 | 0.7295 | 0.7054 |
| lower macro-F1 | 0.5886 | 0.5496 |
| color mean F1 | 0.6591 | 0.6275 |
| upper Acc | 0.8430 | 0.8351 |
| lower Acc | 0.8335 | 0.8241 |

解释：

- 颜色分类在较早阶段已经达到峰值。
- 后续训练综合分数仍可能因为分割或其他 checkpoint 选择逻辑继续变化。
- 下装颜色比上装更容易过拟合或退化，final 相比 best 的 lower macro-F1 下降约 0.039。
- rare colors，如 purple、pink、orange，在两个 head 中都更困难。

这部分适合强调实验意识：

> 不只看最后一个 checkpoint，而是分析不同任务的峰值是否同步。多任务模型可能出现分割指标继续提升、颜色指标开始回落的情况，因此需要根据业务目标选择 checkpoint。

### 7.6 RGB-only 最优实验

配置文件：

```text
experiment/PaddleSeg/experiments/clothes_color_rgb_only/configs/optC_scratch_rgb.yml
```

关键配置：

| 项 | 值 |
| --- | --- |
| 输入 | RGB 3 通道 |
| batch size | 128 |
| iters | 50,000 配置，最佳点出现在 22,000 |
| LR | 0.0075 |
| warmup | 1,000 |
| optimizer | SGD momentum 0.9 |
| weight decay | 5e-4 |
| train split | 71,345 |
| val split | 7,927 |
| resize | 256 |
| color loss | CE，包含 unknown 类 |

最终验证集结果：

| 指标 | RGB-only OptC |
| --- | ---: |
| mIoU | 0.6607 |
| Acc | 0.9453 |
| Kappa | 0.8908 |
| Dice | 0.7721 |
| upper Acc | 0.8222 |
| upper macro-F1 | 0.7730 |
| upper macro-Precision | 0.7915 |
| upper macro-Recall | 0.7589 |
| lower Acc | 0.8192 |
| lower macro-F1 | 0.6772 |
| lower macro-Precision | 0.7353 |
| lower macro-Recall | 0.6864 |
| combined score | 0.6929 |

按分割类别 IoU：

| 类别 | IoU |
| --- | ---: |
| background | 0.9343 |
| upper | 0.8460 |
| lower | 0.8555 |
| dress | 0.4086 |
| hat | 0.6032 |
| glasses | 0.2857 |
| mask | 0.5264 |
| hair | 0.8263 |

#### 7.6.1 RGB-only 配置为什么能赢

RGB-only 最优配置不是简单把 pose 通道删掉，而是做了几处配套调整：

| 调整 | 目的 |
| --- | --- |
| train 数据从 46,878 扩到 71,345 | 弥补去掉 pose prior 后的结构信息，提升颜色覆盖 |
| LR 从 0.005 提到 0.0075 | 更大数据下梯度更稳定，可以使用更高学习率 |
| 增加 1000 iter warmup | 避免前期高 LR 造成训练震荡 |
| weight decay 从 4e-4 到 5e-4 | 更强正则，缓解长训练过拟合 |
| end_lr 设为 1e-6 | 避免 polynomial decay 后期完全进入 LR 死区 |
| `ce_ignore_index=-1` | unknown 类也参与训练，颜色体系从 11 类扩到完整 12 类 |
| `upper_color_weight=0.6`、`lower_color_weight=0.4` | 上下装颜色监督分开加权，降低下装噪声/长尾对训练的冲击 |

这说明最终提升来自组合优化：

```text
更大数据 + 更简单输入 + 更合理 LR schedule + unknown 显式建模 + 上下装 loss 分权
```

#### 7.6.2 RGB-only 的 per-class 改进

`val_comparison.md` 中一个重要细节是：PosePrior 和 ZeroPose 的颜色指标统计 11 类，不含 unknown；RGB-only 统计完整 12 类，包含 unknown。因此 RGB-only 的 macro-F1 实际更严格。

RGB-only 对稀有色尤其有价值。下装颜色 Precision 中：

| 下装颜色 | PosePrior | ZeroPose | RGB-only |
| --- | ---: | ---: | ---: |
| yellow | 0.600 | 0.000 | 0.751 |
| purple | 0.301 | 0.000 | 0.992 |
| pink | 0.475 | 0.000 | 0.648 |
| orange | 0.100 | 0.000 | 0.333 |

这说明：

- ZeroPose 对部分下装稀有色几乎不敢预测。
- PosePrior 能预测一些稀有色，但 Precision 不稳定。
- RGB-only 借助更大训练数据和更完整标签体系，让 rare color 从“几乎不可用”变成“至少可被模型识别”。

需要注意：

- `purple=0.992` 这类极高 precision 可能受样本量影响，不能单独夸大。
- 但多个稀有色整体从 0 或很低提升，方向是可信的。

### 7.7 Pose prior / Zero-pose / RGB-only 对比

`clothes_color_rgb_only/val_comparison.md` 给出了关键对比：

| 方案 | mIoU | Acc | upper F1 | lower F1 | combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| PosePrior OptC | 0.6402 | 0.9428 | 0.7054 | 0.5496 | 0.6375 |
| ZeroPose OptB | 0.6397 | 0.9414 | 0.6262 | 0.4182 | 0.5829 |
| RGB-only OptC | 0.6607 | 0.9453 | 0.7730 | 0.6772 | 0.6929 |

结论：

- RGB-only 最终最优。
- Pose prior 相比 zero-pose 对颜色有帮助，尤其上装颜色，但不如扩大数据和优化训练策略的收益。
- 数据规模、标签质量、颜色类别处理，比手工 pose prior 更关键。

### 7.8 实验经验总结

可复盘的经验：

1. 分割预训练不一定让颜色任务最优  
   预训练模型更偏向保留分割能力，但颜色分类需要全局区域颜色特征，直接加颜色 head 容易欠适配。

2. 从零联合训练在多任务上更自然  
   Option C 让 backbone、decoder、seg head、color head 共同适配，颜色 macro-F1 提升明显。

3. RGB-only 的工程优势很大  
   不依赖 keypoint，不需要 pose cache，不受关键点失败影响，部署链路短，最终指标也更好。

4. 下装颜色比上装更难  
   下装黑/蓝占比极高，其他颜色少，macro-F1 更能暴露长尾问题。

5. 小目标分割决定属性边界  
   glasses、mask、hat 的 IoU 明显低于 upper/lower/hair，需要像素占比分析和 badcase 挖掘。

## 8. 评估与统计：statistics

`statistics` 目录用于从整体指标进一步下钻到属性检出率和 badcase。

关键文件：

```text
statistics/predict_detection.py
statistics/gen_detection_html.py
statistics/per_image_detection.csv
statistics/per_image_detection_train.csv
statistics/detection_analysis_report.html
statistics/badcase/
```

### 8.1 预测统计

`predict_detection.py` 做的事情：

- 使用 RGB-only `iter_22000` checkpoint 对验证集推理。
- 对每张图统计每个属性的 GT/Pred 像素数。
- 计算 GT/Pred 像素占比。
- 判断每类属性的 TP、FP、FN。
- 记录每张图的 mIoU、Acc。
- 输出 badcase 图像、mask 和 meta 信息。

验证集统计文件：

| 文件 | 行数 |
| --- | ---: |
| `per_image_detection.csv` | 7,928 行，含表头，即 7,927 张图 |
| `per_image_detection_train.csv` | 61,271 行 |

### 8.2 HTML 可视化报告

`gen_detection_html.py` 生成：

```text
statistics/detection_analysis_report.html
```

当前报告重点分析：

```text
hat, glasses, mask, hair
```

分析内容：

- 每类属性的 Precision、Recall、F1。
- GT 像素占比分桶后的检出表现。
- Pred 像素占比和 FP 关系。
- badcase 展示。
- 小目标属性漏检/误检模式。

### 8.3 为什么要做像素占比分析

仅看全局 mIoU 会掩盖很多问题：

- upper/lower 像素大，容易主导总体指标。
- glasses/mask 像素很小，错一个就可能对用户感知明显，但对整体 mIoU 影响小。
- 小目标属性更依赖分辨率、清晰度、遮挡情况。

像素占比分析可以回答：

- 一个属性区域占整图多少像素时，模型开始稳定检出？
- 哪些类别在低面积时 precision/recall 崩得最快？
- badcase 是漏检为主，还是误检为主？
- 部署时是否需要对过小属性或过小 crop 做拒答？

### 8.4 badcase 分析价值

`statistics/badcase/meta.json` 记录了 FP/FN 总数和对应样本信息。

badcase 用于：

- 查找标签错误。
- 查找模型系统性误检，例如头发和帽子、口罩和阴影、眼镜和反光。
- 设计数据补充策略。
- 设计拒答规则或后处理规则。

## 9. 拒答策略：refuse-answer

`refuse-answer` 是项目中很适合求职讲述的一部分，因为它从“模型能不能预测”推进到“什么时候不应该预测”。

路径：

```text
refuse-answer/
├── smallcrop/
└── multicolor/
```

## 10. 小图拒答：smallcrop

目标：

> 当 person crop 太小、太模糊或颜色区域像素不足时，不强行输出颜色，改为只输出分割 mask 或降低颜色置信度。

### 10.1 分析方法

`smallcrop` 的 README 和脚本完成了：

- 读取 53,269 张 train+val 图像尺寸。
- 统计 crop 面积、短边、宽高比。
- 按面积分桶，观察分割 mIoU 和颜色 precision 的变化。
- 扫描 short edge 和 area threshold。
- 采样 badcase。

### 10.2 数据分布

关键观察：

- crop 面积中位数约 66k 像素，约等于 180 x 360。
- 40.8% 样本面积小于 50k 像素。
- 64% 样本短边小于 256 px。
- 宽高比中位数约 0.438，说明大多数是竖向人像 crop。
- 唯一尺寸多达 44,077 种，真实数据尺寸分布很散。

### 10.3 性能与尺寸关系

观察：

- 很小 crop 的 mIoU 约 0.306。
- 大 crop 的 mIoU 约 0.458。
- 很小 crop 的颜色 precision 约 0.733。
- 大 crop 的颜色 precision 约 0.947。

结论：

- crop 越大，分割和颜色判断越稳定。
- 颜色比纯分割更敏感，因为颜色分类依赖足够多的有效服装像素。
- 尺寸不是唯一因素，来源域和标签质量也有明显影响。

### 10.4 推荐阈值

部署规则：

```text
if short_edge < threshold_short or area < threshold_area:
    skip color prediction
    output segmentation only
```

阈值方案：

| 策略 | short edge | area | 过滤比例 | 适用场景 |
| --- | ---: | ---: | ---: | --- |
| 保守 | >= 80 | >= 15k | 约 5.6% | 尽量多给结果 |
| 平衡，推荐 | >= 96 | >= 25k | 约 9.9% | 质量与覆盖折中 |
| 激进 | >= 128 | >= 40k | 约 22.0% | 更重视颜色准确 |

推荐上线优先使用平衡策略：

```text
short_edge >= 96 and area >= 25,000
```

### 10.5 面试可讲点

- 分类模型并不应该在所有输入上都给硬答案。
- 拒答策略可以用数据驱动，而不是拍脑袋阈值。
- 尺寸阈值是简单、可解释、低成本的部署保护。
- 小图拒答能显著减少用户最容易感知的颜色错误。

## 11. 多色拒答：multicolor

目标：

> 对多色、条纹、格纹、强花纹衣物，不强行输出单一颜色，避免“把复杂颜色压成一个错误主色”。

### 11.1 v1：Lab 颜色峰值 + RLE 纹理

设计文件：

```text
refuse-answer/multicolor/v1/DESIGN.md
```

方法：

1. 从分割 mask 中取上装或下装像素。
2. 转到 Lab 空间。
3. 在 a*b* 平面上做 128 x 128 直方图。
4. Gaussian smoothing。
5. 检测局部峰值。
6. 合并距离近的颜色峰。
7. 计算 purity。
8. 对像素聚类结果做行/列 RLE，判断条纹/格纹。

核心规则：

- `n_peaks <= 1` 或 `purity > 0.85`：认为接近纯色。
- 多峰但无明显纹理：多色，惩罚更高。
- 条纹/格纹：多色，但可给相对温和惩罚。

v1 结果：

| 指标 | 值 |
| --- | ---: |
| pure set acc | 0.7429 |
| multi set acc | 0.5081 |
| overall accuracy | 0.6222 |
| precision | 0.6763 |
| recall | 0.5081 |
| specificity | 0.7429 |
| F1 | 0.5802 |

### 11.2 v2：纯色门控

设计转变：

> 不再试图完整证明“这是多色”，而是证明“这足够纯，可以交给单色分类器”。如果不能证明纯色，就视为拒答风险。

关键规则：

- 使用像素面积比例，而不是直方图峰高。
- 关注 `main_ratio`、`second_ratio`、`minor_total`、`n_effective_colors`。
- 增加中性色 L 通道分支，解决黑/白/灰图案在 a*b* 空间不明显的问题。

主要阈值：

```text
MIN_ROI_PIXELS = 50
MAIN_RATIO_PURE_MIN = 0.80
SECOND_RATIO_PURE_MAX = 0.10
MINOR_TOTAL_PURE_MAX = 0.20
```

v2 结果：

| 指标 | 值 |
| --- | ---: |
| accuracy | 0.5876 |
| pure recall | 0.5328 |
| multi recall | 0.6364 |
| pure precision | 0.5659 |
| multi precision | 0.6049 |

### 11.3 v3：adaptive Lab

v3 目标：

- 改进 Lab 聚类。
- 提升对多色样本的召回。
- 降低固定阈值对光照变化的敏感性。

v3 结果：

| 指标 | 值 |
| --- | ---: |
| accuracy | 0.6667 |
| pure recall | 0.5109 |
| multi recall | 0.8052 |
| pure precision | 0.7000 |
| multi precision | 0.6492 |

特点：

- 多色召回明显提升。
- 代价是纯色召回偏低，即部分纯色因为阴影/褶皱被误判为多色风险。

### 11.4 v4：shadow merge

设计文件：

```text
refuse-answer/multicolor/v4/IMPLEMENTATION_PLAN.md
refuse-answer/multicolor/v4/config_v4.py
refuse-answer/multicolor/v4/detector_v4.py
```

目标：

- 减少纯色衣物因为高光、阴影、褶皱被拆成多个 Lab cluster 的情况。
- 对同一颜色的明暗变化做合并。
- 保持黑/白/灰等中性色差异的敏感性。

关键规则：

- chromatic cluster 使用 a*b* 和 hue 接近性合并阴影。
- neutral cluster 保留 L 通道差异，避免把黑白灰图案错误合并。
- ROI 小于 50 像素时直接视为高风险。

v4 结果：

| 指标 | 值 |
| --- | ---: |
| accuracy | 0.6598 |
| pure recall | 0.5182 |
| multi recall | 0.7857 |
| pure precision | 0.6827 |
| multi precision | 0.6471 |

### 11.5 多色拒答总结

当前多色拒答不是最终完美模型，但已经形成了清晰方向：

- 对部署而言，误把多色当纯色比误拒一些纯色更危险。
- v3/v4 倾向高多色召回，适合做 conservative gate。
- 规则模型可解释，便于调阈值和分析 badcase。
- 后续可以用人工标注多色集训练轻量二分类器，规则特征作为先验或特征输入。

## 12. 关键实验结论汇总

### 12.1 最重要结论

1. RGB-only 最优，不依赖 pose prior。
2. 数据量和标签质量比额外结构先验更关键。
3. 从零联合训练比“分割预训练 + 颜色头”更适合颜色任务。
4. 颜色任务必须看 macro-F1，不能只看 accuracy。
5. 下装颜色是主要难点，源于类别长尾和遮挡。
6. 小目标属性需要单独做像素占比分析。
7. 部署需要拒答策略，尤其是小图和多色衣物。

### 12.2 最终推荐模型

推荐使用：

```text
RGB-only PP-LiteSeg-STDC2 OptC scratch
checkpoint: iter_22000
```

原因：

- 指标最好。
- 工程链路最简单。
- 不依赖 pose 检测，减少推理耗时和失败点。
- 在上装、下装颜色 macro-F1 上都明显优于 pose prior 和 zero-pose。

### 12.3 仍然薄弱的类别

分割薄弱类别：

- glasses：IoU 0.2857
- dress：IoU 0.4086
- mask：IoU 0.5264

原因推测：

- 像素区域小。
- 标注边界更难。
- 遮挡和反光影响明显。
- 类别定义存在歧义，例如帽子和头发、口罩和脸部阴影、眼镜和高光。

颜色薄弱点：

- 下装稀有色。
- 多色/花纹/条纹。
- 低分辨率 crop。
- 光照导致的黑/灰/蓝、白/灰、棕/橙混淆。

## 13. Tricks 与工程细节

这一节适合面试时展开，体现你不是只跑训练，而是处理了大量真实工程问题。

### 13.1 数据层面

| Trick | 解决的问题 |
| --- | --- |
| dHash 近重复去重 | 连续帧和重复 crop 导致训练集泄漏或过拟合 |
| 按来源/前缀分桶去重 | 降低近重复检测复杂度 |
| 稀有色数据补齐 | 缓解颜色长尾，提升 macro-F1 |
| Qwen-VL 结构化标注 | 降低人工颜色标注成本 |
| SAM3 自动 mask | 快速生成多属性分割监督 |
| 人工 review HTML | 发现系统性标注错误 |
| 真实场景 crop 补充 | 缩小公开数据和业务场景 domain gap |

### 13.2 训练层面

| Trick | 解决的问题 |
| --- | --- |
| 多任务共享 backbone | 同时学习区域语义和颜色语义 |
| 双颜色 head | 避免上装/下装颜色混淆 |
| OHEM segmentation loss | 增强小目标/困难像素学习 |
| RGB-only 最终方案 | 简化部署并提升效果 |
| unknown 类显式建模 | 增强不确定颜色表达 |
| BGR/RGB 转换校验 | 避免隐性输入通道错误 |
| 颜色增强只作用于 RGB | 避免破坏 pose prior 通道 |
| pose/label cache | 加速数据加载和多轮训练 |

### 13.3 评估层面

| Trick | 解决的问题 |
| --- | --- |
| macro-F1 | 避免大类 accuracy 掩盖长尾失败 |
| per-class IoU | 发现 glasses/mask/dress 等薄弱类别 |
| 像素占比分桶 | 分析小目标属性何时可稳定检出 |
| badcase HTML | 快速定位误检/漏检样本 |
| smallcrop threshold scan | 用数据驱动拒答阈值 |
| multicolor pure gate | 避免复杂衣物被强行单色化 |

## 14. 面试可讲的 STAR 版本

### 14.1 背景 Situation

项目需要识别人像中的服装区域和颜色，但真实数据存在遮挡、低清、小目标、颜色长尾、多色衣物、自动标注噪声等问题。单纯训练一个分类模型无法满足稳定部署。

### 14.2 任务 Task

负责搭建从数据生产到模型训练评估的完整流程，提升人体属性分割和上下装颜色识别效果，并为低置信场景设计拒答策略。

### 14.3 行动 Action

- 整合 LIP、IPC、UPAR、真实场景 crop 等多源数据，统一 8 类分割和 12 类颜色标签体系。
- 使用 SAM3 自动生成分割 mask，使用 Qwen-VL 自动标注颜色，并通过 review HTML 和清洗脚本控制质量。
- 基于 PaddleSeg 改造 PP-LiteSeg-STDC2，增加上装和下装两个颜色 head，构建分割 + 颜色多任务学习。
- 对比 pose prior、zero-pose、RGB-only、分阶段训练和从零联合训练，选择最优方案。
- 建立按类别、按像素占比、按 badcase 的评估体系。
- 针对小 crop 和多色衣物设计拒答策略。

### 14.4 结果 Result

- 最优 RGB-only 模型在验证集达到 `mIoU 0.6607`、上装 `macro-F1 0.7730`、下装 `macro-F1 0.6772`、综合分数 `0.6929`。
- 发现并验证 RGB-only + 更大数据优于 pose prior 方案，简化了部署链路。
- 建立小图拒答推荐阈值 `short_edge >= 96` 且 `area >= 25,000`，过滤约 9.9% 高风险样本。
- 多色拒答 v3/v4 将多色召回提升到约 0.79 到 0.81，适合做保守风险门控。

## 15. 后续改进方向

### 15.1 数据改进

- 针对下装黄色、紫色、橙色继续采样补齐。
- 对 glasses、mask、dress 做专项 badcase 清洗和补标。
- 建立多色/条纹/格纹人工标注集，用于训练更稳定的二分类器。
- 将真实场景线上数据按失败模式回流，做主动学习。

### 15.2 模型改进

- 引入 mask-aware color pooling：只在预测或 GT 服装区域内聚合颜色特征。
- 尝试轻量 attention，让颜色 head 更关注上装/下装区域。
- 对颜色分类使用 class-balanced loss、focal loss 或 logit adjustment。
- 对分割小目标类别增加 class weight 或 boundary loss。
- 尝试更强 backbone，但需要权衡部署速度。

### 15.3 评估改进

- 建立统一固定 test set，避免不同实验 split 不一致导致比较困难。
- 对颜色混淆矩阵做定期报告。
- 给小图、多色、遮挡、夜间/低光等维度单独建 slice 指标。
- 把拒答后的 coverage/accuracy 曲线纳入主评估，而不是只评估强制输出。

### 15.4 部署改进

- 输出颜色时同时输出置信度、是否拒答、拒答原因。
- 对小图、多色、低置信分类分别给不同 refuse code。
- 将 RGB-only 模型和拒答规则串成统一 inference pipeline。
- 对真实流量持续记录低置信样本，形成增量数据闭环。

## 16. 项目局限与注意事项

1. 不同实验的 split 不完全一致  
   pose prior 系列使用 46,878/6,391，RGB-only 使用 71,345/7,927。对比趋势可信，但严格论文式比较需要统一 split。

2. 自动标注存在噪声  
   SAM3 mask 和 Qwen 颜色标签都需要人工抽检。项目中已经有 review 工具，但后续仍应持续清洗。

3. 多色拒答仍是规则原型  
   v3/v4 多色召回较高，但纯色召回偏低。适合作为保守 gate，不宜包装成已经完全解决的问题。

4. 小图阈值来自统计规律  
   `short_edge >= 96 and area >= 25k` 是当前数据上的平衡选择，上线到新摄像头/新业务域后应重新校准。

5. 部分类别天然难  
   glasses、mask、dress 的像素少、边界复杂、标注噪声高，需要专项优化。

6. 训练集颜色长尾明显  
   下装黑/蓝占比过高，accuracy 不能真实反映稀有色效果，必须同时看 macro-F1。

## 17. 面试问答准备

### Q1：为什么最后 RGB-only 比 pose prior 更好？

可以回答：

> 一开始假设 pose prior 能帮助模型区分上半身和下半身，所以做了 RGB+pose 6 通道方案。但通过 zero-pose ablation 发现，pose prior 对分割 mIoU 帮助不大，对颜色有一定帮助但有限。后续 RGB-only 使用更大、更干净的数据和更合理的联合训练策略，最终指标超过 pose prior。这个结果说明在该任务里，数据规模和标签质量比手工结构先验更关键，同时 RGB-only 部署更简单。

### Q2：为什么不用 accuracy 作为颜色主指标？

可以回答：

> 因为颜色类别非常长尾，尤其下装黑色和蓝色占比超过 60%。如果只看 accuracy，模型偏向大类也能得到不错结果，但稀有颜色可能完全失败。因此我更关注 macro-F1、macro-precision、macro-recall，它们能更公平地反映每个颜色类别的效果。

### Q3：项目中最大的工程难点是什么？

可以回答：

> 最大难点不是单次训练，而是数据闭环。需要从视频抽帧、行人裁剪、自动分割、颜色标注、去重清洗、人工审核、split 构建到训练评估全部串起来。任何一个环节有噪声都会影响颜色任务，尤其是长尾颜色和小目标属性。所以项目里做了 dHash 去重、SAM3/Qwen 自动标注、review HTML、badcase 分析和拒答策略。

### Q4：小图拒答为什么重要？

可以回答：

> 颜色分类依赖足够多的有效服装像素。统计发现 crop 越小，分割 mIoU 和颜色 precision 都明显下降。与其在低质量输入上强行输出错误颜色，不如用 short edge 和 area 阈值拒答或只输出分割。推荐阈值是 short edge 大于等于 96 且面积大于等于 25k，过滤约 9.9% 高风险样本。

### Q5：多色拒答为什么不用直接分类？

可以回答：

> 当前多色样本标注规模有限，直接训练分类器容易过拟合。因此先做可解释规则：从服装 mask 内取像素，在 Lab 空间聚类，计算主色比例、次色比例和小颜色总比例；如果不能证明足够纯色，就作为拒答风险。后续如果有更多人工标注，可以把这个规则升级为轻量二分类模型。

## 18. 关键文件索引

### 数据

```text
data/LIP_clothes_accessory_unified/summary.json
data/splits/train.txt
data/splits/val.txt
data/splits-new/train.txt
data/splits-new/val.txt
data/UPAR_rare_color/
data/object_detection_0309-0429/
data/all-data-clean-image.txt
color_label_distribution_train.md
```

### 额外数据

```text
extradata/scripts/extract_frames_batch.py
extradata/scripts/qwen_color_annotate.py
extradata/scripts/qwen_color_clean.py
extradata/upar-dataset/rare_color_samples.txt
```

### 标注

```text
labeling/annotate_ipc.py
labeling/qwen_upper_lower_color.py
labeling/qwen_upper_lower_color_prompt.txt
labeling/server.py
labeling/generate_annotation_review.py
```

### 训练与实验

```text
experiment/PaddleSeg/experiments/pose_prior_stdc2/
experiment/PaddleSeg/experiments/pose_prior_stdc2/README.md
experiment/PaddleSeg/experiments/pose_prior_stdc2/pose_prior_dataset.py
experiment/PaddleSeg/experiments/pose_prior_stdc2/tools/convert_stdc2_pretrained_3ch_to_6ch.py
experiment/PaddleSeg/experiments/clothes_accessory_poseprior_stdc2/
experiment/PaddleSeg/experiments/clothes_accessory_poseprior_stdc2/README.md
experiment/PaddleSeg/experiments/clothes_accessory_poseprior_stdc2/clothes_accessory_poseprior_dataset.py
experiment/PaddleSeg/experiments/clothes_color_poseprior_stdc2/
experiment/PaddleSeg/experiments/clothes_color_poseprior_stdc2/clothes_color_model.py
experiment/PaddleSeg/experiments/clothes_color_poseprior_stdc2/clothes_color_poseprior_dataset.py
experiment/PaddleSeg/experiments/clothes_color_poseprior_stdc2/losses.py
experiment/PaddleSeg/experiments/clothes_color_poseprior_stdc2/train_color_poseprior.py
experiment/PaddleSeg/experiments/clothes_color_rgb_only/
experiment/PaddleSeg/experiments/clothes_color_rgb_only/dataset_rgb.py
experiment/PaddleSeg/experiments/clothes_color_rgb_only/train_color_rgb.py
experiment/PaddleSeg/experiments/clothes_color_rgb_only/configs/optC_scratch_rgb.yml
experiment/PaddleSeg/experiments/clothes_color_zeropose_ablation/
experiment/PaddleSeg/experiments/clothes_color_zeropose_ablation/dataset_zeropose.py
experiment/PaddleSeg/experiments/clothes_color_zeropose_ablation/train_color_zeropose.py
experiment/PaddleSeg/experiments/clothes_color_poseprior_stdc2/THREE_PLANS_ANALYSIS.md
experiment/PaddleSeg/experiments/clothes_color_rgb_only/val_comparison.md
experiment/PaddleSeg/experiments/clothes_color_zeropose_ablation/analysis.md
experiment/PaddleSeg/experiments/pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256_optC_scratch/color_metrics_report.md
experiment/PaddleSeg/experiments/pp_liteseg_stdc2_clothes_color_2head_rgb_3ch_256_optC_scratch/train_nohup.log
experiment/PaddleSeg/experiments/pp_liteseg_stdc2_clothes_color_2head_rgb_3ch_256_optC_scratch/val_comparison.md
```

### 评估

```text
statistics/predict_detection.py
statistics/gen_detection_html.py
statistics/per_image_detection.csv
statistics/detection_analysis_report.html
statistics/badcase/
```

### 拒答

```text
refuse-answer/smallcrop/README.md
refuse-answer/smallcrop/analyze_and_report.py
refuse-answer/multicolor/v1/DESIGN.md
refuse-answer/multicolor/v2/RULES.md
refuse-answer/multicolor/v3/
refuse-answer/multicolor/v4/
```

## 19. 一页版总结

`segment-color` 是一个完整的人体属性识别项目，核心任务是同时做人像 8 类属性分割和上下装 12 类颜色分类。项目不仅训练模型，还覆盖了真实视觉项目中最消耗时间的数据工程：视频抽帧、YOLO 行人裁剪、SAM3 自动分割、Qwen-VL 颜色标注、人工审核、dHash 去重、稀有色补齐、真实场景数据扩充。

模型侧基于 PaddleSeg 改造 PP-LiteSeg-STDC2，增加上装和下装双颜色 head，形成分割 + 颜色多任务网络。实验中系统对比了 RGB+pose prior、zero-pose、RGB-only，以及分割预训练、分阶段训练、从零联合训练等方案。最终发现 RGB-only + 更大数据 + 从零联合训练效果最好，验证集达到 `mIoU 0.6607`、上装颜色 `macro-F1 0.7730`、下装颜色 `macro-F1 0.6772`，综合分数 `0.6929`。

评估侧不仅看整体 mIoU 和 accuracy，还做了 per-class IoU、颜色 macro-F1、像素占比 vs 检出率、badcase HTML 分析，发现 glasses、mask、dress 等小目标/难类别仍是瓶颈。部署侧进一步设计了小图拒答和多色拒答，推荐小图阈值为 `short_edge >= 96` 且 `area >= 25,000`，多色拒答则通过 Lab 聚类、主色比例和 shadow merge 判断衣物是否足够纯色。

从求职角度，这个项目的亮点不是单一模型指标，而是完整的数据闭环和工程化思维：能从原始视频和公开数据出发，构建训练数据，发现数据问题，设计模型实验，建立评估体系，再把评估结果转化为部署拒答策略。
