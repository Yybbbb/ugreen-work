# Florence-2 人物区域描述项目实习工作总结

> 本文基于 `florence-caption/` 与 `florence-data/` 两个目录中的代码、实验文档、数据统计、模型产物和评测报告整理。项目文件时间主要集中在 **2026 年 7 月 6 日至 7 月 17 日**。由于 `florence-caption/` 当前没有有效 Git commit，`florence-data/` 也不是 Git 仓库，本文对个人工作的梳理以目录内现有产物和文件时间线为依据，而不是以提交记录为依据。

## 项目背景：Florence-2 模型选型

项目需要识别和描述的人物属性具有明显的开放集特征，例如服装类型、颜色、发型、配饰、手持物、动作和人物交互等。相比将任务建模为固定标签集合上的纯 ViT 分类问题，我选择使用视觉语言模型（VLM）：VLM 可以直接生成自然语言形式的开放集属性，不需要提前穷举和维护完整标签范围，也不需要为每个新增属性重新设计类别和补充分类标注数据；当图像信息不足、人物区域过小或属性不可见时，模型还可以通过“不确定”或不描述该属性的方式实现一定程度的拒答，降低强制分类带来的错误预测。

考虑业务部署成本和推理效率，模型规模目标控制在约 **0.2B～0.5B**。前期对 Florence-2、SmolVLM、InternVL 等轻量视觉语言模型进行了能力对比，重点考察人物区域理解、细粒度属性覆盖、bbox 条件描述、输出稳定性和本地部署成本。实际测试中，Florence-2 在该人物区域描述任务上的综合表现最好，同时原生支持 loc token 和 region-level 视觉语言任务，便于将人物 bbox 直接编码进 prompt，并继续扩展单区域和多区域描述能力，因此最终选择 Florence-2 作为项目基础模型。

## 初版技术路线：SFT 与 DPO

项目初版首先采用监督微调（SFT）建立稳定的人物描述能力。训练目标由 Qwen 辅助构造，描述长度控制在约 **15 个英文单词**，在保证人物核心属性覆盖的同时限制输出冗余。除描述内容外，还对输出语法、句式和属性组织顺序进行约束：描述以人物主体开头，随后优先描述服装及颜色，再补充眼镜、背包、手持物等配饰或随身物品，以及必要的姿态和动作信息。通过统一的数据表达风格，降低同一属性的句式波动，使 Florence-2 更容易学习稳定、紧凑且适合下游解析的人物描述格式。

在 SFT 基础上，进一步使用直接偏好优化（DPO）优化描述质量和格式遵循能力。正样本由 Qwen 按目标语法、句式、长度和属性顺序生成；负样本则不采用简单随机扰动，而是先对 Florence-2 的实际 Bad Case 进行归纳，例如主体缺失、属性顺序混乱、颜色与服装绑定错误、配饰遗漏、描述重复、过度描述背景、句式不完整或描述了错误人物等，再将这些错误规律交给 Qwen，有针对性地生成具有迷惑性但质量较差的 rejected response。通过这种“真实 Bad Case 总结—错误模式抽象—Qwen 构造偏好对”的方式，使 DPO 数据更贴近模型实际缺陷，引导模型偏好主体明确、属性准确、结构稳定且信息密度更高的描述。

## 一、工作概述

实习期间，我围绕 **Florence-2 在人物区域细粒度描述场景中的数据构建、模型微调、推理加速、质量评测和工程交付** 开展了一套完整工作。

项目的核心目标包括：

1. 从真实业务图像中检测人物并构建高质量人物区域描述数据集；
2. 将 Florence-2 原本“一个 bbox 推理一次”的单区域描述方式扩展为“多个 bbox 单次推理”的多区域描述任务；
3. 系统解决多区域输出中的描述重复、区域错配、顺序敏感和长序列稳定性问题；
4. 通过数据清洗、图像去重和按相似簇划分训练测试集，降低数据污染与评测泄漏；
5. 针对高质量人物属性描述重新训练单区域模型，显著提升描述完整性和文本匹配指标；
6. 建立训练、推理、评测、Bad Case 分析、Web Demo 和模型交付的完整工程链路。

从工作性质看，我承担的并非单一模型训练任务，而是覆盖了以下环节：

- **数据工程**：人物筛选、YOLO 检测、区域裁剪、描述生成、规则清洗、精确去重、感知哈希去重；
- **任务设计**：Florence-2 新任务 token、输入输出协议、bbox 编码方式与后处理方案；
- **模型训练**：多区域任务和单区域人物描述任务的微调与参数实验；
- **实验研究**：多种输出格式、crop 排序方式、训练数据版本和推理策略对比；
- **评测体系**：内容匹配、区域归属、loc token 回显、输出数量、文本指标、属性覆盖和速度评测；
- **工程交付**：CLI 推理、批量推理、自动训练评测脚本、Web Demo、checkpoint 打包和使用文档。

## 二、两个目录的职责划分

### 2.1 `florence-data/`：人物区域数据生产与治理

该目录承担上游数据处理工作，主要内容包括：

- 使用 Qwen 对候选人物图像进行“完整人物”筛选；
- 使用 YOLO26m 对图片进行人物检测和人物区域裁剪；
- 调用 Florence-2 为每个人物区域生成描述；
- 对 Qwen 生成的人物描述进行规则化清洗；
- 分析完全重复和近重复描述；
- 基于文本、bbox 和图像 dHash 识别相似帧与相似 crop；
- 构建去重后的训练数据集及相似簇清单；
- 生成可视化样例和人工复核材料。

主要脚本包括：

| 脚本 | 作用 |
|---|---|
| `florence-data/scripts/check_complete_person_qwen.py` | 使用 Qwen 判断候选图像中的人物是否完整、是否适合作为训练样本 |
| `florence-data/scripts/make_qwen_cleaning_review_md.py` | 从筛选结果抽样并生成 Markdown 可视化复核页 |
| `florence-data/scripts/crop_qwen_persons_yolo26m.py` | 使用 YOLO26m 检测人物，生成 bbox、crop 和结构化 JSON |
| `florence-data/scripts/clean_region_descriptions_qwen.py` | 使用 Qwen 清洗人物区域描述，统一描述质量和表达方式 |
| `florence-data/scripts/dedup_region_descriptions_exact.py` | 按描述文本进行全局精确去重 |
| `florence-data/scripts/analyze_scene_frame_crop_similarity.py` | 综合文本相似度、bbox IoU 和图像 dHash 分析相似帧及相似人物区域 |
| `florence-data/scripts/build_dhash_dedup_dataset.py` | 根据相似帧聚类结果保留代表帧，生成 dHash 去重数据集 |
| `florence-data/scripts/dup_analysis.py` | 分析重复描述模式及典型重复问题 |
| `florence-data/scripts/dup_visualize.py`、`dup_visualize2.py` | 将重复、近重复样本绘制为可视化图片，支持人工核查 |

### 2.2 `florence-caption/`：模型训练、推理、评测与交付

该目录承担下游 Florence-2 任务开发工作，主要内容包括：

- 管理 Florence-2 基础模型及已有人物区域描述 checkpoint；
- 扩展 `<REGIONS_TO_DESCRIPTIONS>` 多区域描述任务；
- 将原始人物区域结果转换成 Florence-2 训练 JSONL；
- 训练多种数据格式、多种区域顺序和多种超参数的模型；
- 开发单区域与多区域推理脚本；
- 建立准确率、匹配率、区域错配和速度评测；
- 进行 Bad Case 分析和改进方案设计；
- 提供浏览器交互 Demo 与独立 checkpoint 交付包。

其中 `multi_region_description/` 是项目实验主体，保存了：

- 17 类训练数据目录；
- 16 组主要 checkpoint 实验；
- 多区域和单区域评测结果；
- 多种自动化训练评测脚本；
- Bad Case 查看器和 Web Demo；
- 速度、描述匹配、crop 顺序、输出长度等专项报告。

## 三、具体工作内容

### 3.1 构建人物图像筛选与检测流水线

#### 工作内容

我首先搭建了人物区域数据的上游生产流程：

1. 使用 Qwen 对候选人物图片进行完整性判断，筛除严重截断、目标不完整或不适合作为人物属性描述训练数据的图片；
2. 生成正负样本可视化复核页，支持人工检查自动筛选质量；
3. 使用 YOLO26m 进行人物检测；
4. 将检测结果转换成统一的 `xyxy` bbox、Florence 0～999 loc 坐标和人物 crop；
5. 保存图片信息、检测信息、crop 路径和后续描述字段，形成结构化 JSON 数据。

#### 数据产出

`qwen-person-json-yolo26m` 数据生产统计显示：

- 选中图片：**3,692 张**；
- 实际进入处理队列：**3,667 张**；
- 成功处理：**3,667 张**；
- 已存在而跳过：**25 张**；
- 失败：**0 张**。

这说明数据处理脚本具备断点续跑和跳过已有结果的能力，并在该批数据上实现了零失败处理。

### 3.2 批量生成 Florence 人物区域描述

#### 工作内容

基于已有 Florence-2 人物区域描述 checkpoint，我开发和整理了人物区域批量描述流程：

- 读取 YOLO 检测生成的结构化 JSON；
- 将 pixel bbox 转换为 Florence 0～999 loc token；
- 支持带 `person` region name 和不带 region name 两种 prompt；
- 对单张图片的多个人物 bbox 逐一生成描述；
- 保存原始输出、清洗后文本、bbox、状态和错误信息；
- 支持分片执行、限制样本数量、限制单图区域数、跳过已完成文件和汇总运行结果。

#### 数据产出

不带 region name 的批量描述数据统计为：

- 输入 JSON：**3,692 个**；
- 成功处理图片：**3,692 张**；
- 来源人物区域：**12,012 个**；
- 成功描述区域：**12,012 个**；
- 区域失败数：**0**；
- 无有效人物区域图片：**378 张**。

此外，我针对同一张包含 12 个人物的图片分别进行带 `person` 和不带 `person` prompt 的复现实验，两种方式均完成 **12/12 区域成功推理**，用于验证 prompt 行为和结果差异。

### 3.3 设计 Florence-2 多区域单次推理任务

#### 背景问题

Florence-2 原有 `<REGION_TO_DESCRIPTION>` 任务每次只处理一个 bbox。同一张图片有 N 个人物时，需要执行 N 次完整推理，视觉编码器也会重复计算 N 次，推理时间随人物数量线性增长。

#### 我的设计

我新增了多区域任务 token：

```text
<REGIONS_TO_DESCRIPTIONS>
```

输入为一张图片和多个 Florence loc bbox：

```text
<REGIONS_TO_DESCRIPTIONS><loc_x1><loc_y1><loc_x2><loc_y2><sep>...
```

我设计并验证了多种输出协议。

#### 方案 A：按顺序输出描述

```text
description1<sep>description2<sep>description3
```

特点：

- 输出短、解析简单；
- 描述与 bbox 依赖输入顺序隐式对应；
- 如果模型漏生成一条描述，后续描述可能整体错位。

#### 方案 B：描述后显式回显 bbox

```text
description1<loc_x1><loc_y1><loc_x2><loc_y2>description2<loc_...>
```

特点：

- 延续 Florence-2 原生 `description + loc` 输出范式；
- 每条描述与 bbox 显式绑定；
- 即使输出顺序变化，也能通过 loc token 重新建立对应关系；
- loc run 自身即可作为分隔符，无须完全依赖 `<sep>`。

#### 进一步实验格式

我还构建并训练了以下格式：

- `description-sep`；
- `description-loc`；
- `description-loc-sep`；
- 带新 prompt 的 Plan B；
- 不同 epoch、warmup 和任务混合配置。

#### Processor 改造

我修改 Florence-2 processor 的任务映射、prompt 构造和后处理逻辑，使其能够：

- 识别 `<REGIONS_TO_DESCRIPTIONS>`；
- 构造多 bbox prompt；
- 解析 `<sep>` 分隔结果；
- 解析严格的“描述 + 4 个 loc token”结构；
- 对描述数量和 bbox 数量进行校验；
- 在异常情况下保留原始输出供定位。

### 3.4 构造多区域训练数据并解决描述重复问题

#### 原始数据问题

多区域任务的数据来自逐 crop 的 Florence 推理结果。分析后发现，同一张图片中的人物描述存在严重重复：

| 重复类型 | 占比 |
|---|---:|
| 全部描述完全相同 | 14.6% |
| 部分描述重复 | 47.7% |
| 全部描述唯一 | 37.7% |

其主要原因包括：

- 小人物区域信息不足，模型生成泛化描述；
- YOLO 对同一人物产生重叠框；
- 同场景人物穿着和动作相似；
- 原模型有时未严格聚焦 bbox，而是描述整张图像。

如果直接使用这些数据训练，多区域模型容易学习“忽略 bbox、重复输出同一句”的错误捷径。

#### 我的处理策略

在 `prepare_data.py` 中，我实现了以下规则：

1. 只保留推理状态正常的 region；
2. 对同一图片内完全相同的描述进行去重；
3. 重复描述只保留 crop 面积最大的区域；
4. 去重后区域数超过 5 时，只保留面积最大的 5 个区域；
5. 无有效 region 的图片直接跳过；
6. 保留 N=1 样本，使多区域模型可以自然退化到单区域输入；
7. 对 bbox 数和描述数执行一致性校验。

#### 数据产出

从 no-name Florence 描述数据构建得到：

- 总样本：**3,314 张图片**；
- 训练集 / 测试集：**2,983 / 331**；
- N=1：1,270 条；
- N=2：568 条；
- N=3：383 条；
- N=4：328 条；
- N=5：765 条；
- 多区域样本描述唯一率：**100%**；
- bbox 与描述数量错配：**0**；
- 无 crop 而跳过：378 张。

### 3.5 开发 Florence-2 微调训练与自动化实验脚本

#### 训练实现

我开发了单 GPU Florence-2 微调入口 `florence-caption/scripts/train_multi_region.py`，主要能力包括：

- 从本地 checkpoint 加载自定义 Florence-2 模型与 processor；
- 支持 fp16 / bf16；
- 支持 batch size、epoch、学习率、warmup、最大序列长度等参数；
- 保存每个 epoch checkpoint 和 final checkpoint；
- 使用小学习率轻量微调，降低新任务训练对已有能力的破坏；
- 支持多任务数据，缓解模型遗忘原有 caption 或 region description 能力。

#### 自动化实验

为提高实验可重复性，我编写了多组 Shell 流水线，将数据准备、训练、推理和评测串联起来，包括：

- Qwen 数据 2～5 人多区域训练；
- 三种输出格式对比训练；
- 不同 crop 顺序训练；
- 清洗数据与未清洗数据对比；
- top-to-bottom 排序实验；
- 单区域 dHash 去重数据训练；
- 训练完成后自动检查 checkpoint 并运行评测。

项目中保留了 **16 组主要 checkpoint 实验**，体现了从方案 A、方案 B、prompt 调整、输出格式调整，到 Qwen 清洗数据和单区域训练的连续迭代过程。

### 3.6 建立多区域描述质量评测体系

仅使用 BLEU 或 ROUGE 无法判断“描述内容基本正确，但分配给了错误 bbox”的问题。因此，我将评测拆分为多个维度。

#### 评测维度

1. **Count Match Rate**：输出描述数量是否等于输入 bbox 数量；
2. **Loc Echo Accuracy**：模型回显的 bbox loc token 是否正确；
3. **Positional Word F1**：按输入位置直接对齐后的文本 Word F1；
4. **Best-Permutation Word F1**：允许重新排列预测描述后的最佳 Word F1；
5. **Content Pass Rate**：描述内容是否达到基本匹配阈值；
6. **Acceptable Region Match Rate**：描述是否可接受地属于当前 bbox；
7. **Confident Region Match Rate**：当前 bbox 的匹配得分是否显著优于其他 bbox；
8. **Wrong BBox Region Rate**：描述内容正确但更符合其他人物区域的比例；
9. **Content Mismatch Rate**：描述本身与所有区域都不匹配的比例；
10. **Fully Acceptable Frame Rate**：一张图中所有人物描述均可接受的比例。

#### 评测方法上的贡献

- 使用小写字母数字 token 的 Word F1 进行文本匹配；
- 构建 bbox 内部候选匹配矩阵；
- 对描述与 GT region 进行最优分配；
- 区分“内容错误”和“区域分配错误”；
- 输出逐帧、逐 crop 和按 crop 数量分组统计；
- 生成错配记录供 Bad Case Viewer 可视化。

这套方法能够更准确地诊断多区域生成模型，而不是仅给出单一平均文本相似度。

### 3.7 研究 crop 输入顺序对区域错配的影响

#### 实验设计

我分别训练并对比了四种人物 bbox 排序方式：

- Random：随机顺序；
- Area Desc：按 bbox 面积从大到小；
- Left-to-Right：按人物中心点从左到右；
- Top-to-Bottom：按人物中心点从上到下。

#### 核心结果

在约 832 张测试图片、2,353 个人物区域上，Left-to-Right 综合效果最佳：

| 指标 | Random | Area Desc | Left-to-Right | Top-to-Bottom |
|---|---:|---:|---:|---:|
| Acceptable Region Match Rate | 75.89% | 88.08% | **89.61%** | 84.32% |
| Confident Region Match Rate | 64.14% | 76.59% | **78.07%** | 73.78% |
| Wrong BBox Region Rate | 19.68% | 8.26% | **7.03%** | 11.98% |
| Content Mismatch Region Rate | 4.43% | 3.66% | **3.36%** | 3.70% |
| Fully Acceptable Frame Rate | 60.53% | 73.65% | **75.69%** | 68.51% |
| Average Positional Word F1 | 0.8053 | 0.8436 | **0.8475** | 0.8337 |

与随机顺序相比，Left-to-Right：

- 可接受区域匹配率提升 **13.72 个百分点**；
- Wrong BBox Region Rate 从 **19.68% 降至 7.03%**，下降 **12.65 个百分点**；
- 完全可接受帧比例从 **60.53% 提升至 75.69%**；
- 错配 crop 数从 462 个降至 165 个。

在 2～5 crop 场景中，Left-to-Right 的可接受匹配率均为四种方案最高；5 crop 场景仍可达到 **87.61%**，而随机顺序下降到 **55.21%**。

#### 实验结论

固定且符合视觉阅读习惯的空间顺序可以显著降低多区域描述错配。模型不仅学习 bbox 内容，也会利用序列位置先验，因此训练和推理必须采用一致的确定性排序策略。

### 3.8 分析 Bad Case 并提出结构性改进方案

#### 主要问题定位

我通过 Bad Case 报告和可视化查看器，将模型问题归纳为：

- 相邻人物属性串位；
- 模型按视觉显著性而非 prompt 顺序输出；
- 多人服装相似时描述归属模糊；
- 输出描述数量缺失造成整体错位；
- loc token 正确但描述主体错误；
- 长序列后部人物描述质量下降；
- 训练数据中的重复和弱区分描述削弱 bbox grounding。

#### 提出的改进方案

1. 在输出中显式绑定 bbox 与描述，优先使用 `description + loc`；
2. 联合随机打乱 prompt 与 label 的 region 顺序，打破固定位置捷径；
3. 混入单区域辅助样本，强化“只描述给定 bbox”的基础能力；
4. 构造相邻人物 hard cases，提高困难样本采样权重；
5. 对数量错误执行 fallback，退回逐 crop 推理；
6. 限制单次 crop 数量或按批次拆分，平衡速度和鲁棒性；
7. 混合原有 `<REGION_TO_DESCRIPTION>` 数据，降低灾难性遗忘。

这些改进不是简单调学习率，而是针对模型输出结构、训练采样和数据难度进行系统优化。

### 3.9 完成多区域推理速度基准测试

#### 初期 Plan A 与逐 crop 推理对比

在 331 张图片、867 个 crop 的测试集上：

- 多区域 Plan A 总耗时：**152.9 秒**；
- 逐 crop 推理总耗时：**182.7 秒**；
- 总体加速：**1.19 倍**；
- 平均单图耗时从 552.1 ms 降至 462.1 ms；
- N=3～5 时加速约为 1.26～1.30 倍。

#### 完整 2～5 crop 顺序执行基准

后续在 **8,324 张图片、23,427 个 crop** 上进行了更完整的多区域与逐区域顺序推理对比：

| Crop 数 | 图片数 | Crop 数 | 生成阶段加速 | 端到端加速 |
|---:|---:|---:|---:|---:|
| 2 | 4,219 | 8,438 | 1.410× | **1.492×** |
| 3 | 2,159 | 6,477 | 1.639× | **1.784×** |
| 4 | 1,218 | 4,872 | 1.802× | **2.007×** |
| 5 | 728 | 3,640 | 1.940× | **2.229×** |
| 总体 | 8,324 | 23,427 | 1.633× | **1.783×** |

总体端到端 crop 吞吐从约 **4.21 crop/s 提升至 7.50 crop/s**。

实验验证了最初的技术判断：人物数量越多，共享一次视觉编码带来的收益越明显；在 5 人场景中，端到端速度达到逐 crop 推理的 **2.23 倍**。

### 3.10 建立数据精确去重与近重复去重流程

#### 全局描述精确去重

对 `qwen_person_2to6_region_descriptions` 进行全局描述文本精确去重，结果为：

- 输入帧：**8,713**；
- 输入 crop：**25,761**；
- 全局唯一描述：**20,287**；
- 删除重复 crop：**5,474**；
- 输出帧：**6,844**；
- 输出 crop：**18,246**。

该流程用于快速识别完全相同的模板化描述，但也暴露出仅按文本精确去重可能导致有效帧被丢弃的问题，因此后续继续开发了更精细的图像与区域联合去重。

#### 文本、bbox 与 dHash 联合相似分析

我设计了场景内相似帧检测算法，综合使用：

- crop 描述文本相似度；
- bbox IoU；
- 整帧图像 dHash 汉明距离；
- 匹配 crop 数量与覆盖率；
- 文本、bbox 和图像的加权帧级相似度。

对每个场景内的帧对进行比较，并通过一对一 crop 匹配和连通分量构建相似帧簇。

#### dHash 去重结果

从清洗后的 Qwen 数据中：

- 原始帧：**8,713**；
- 原始 crop：**25,761**；
- 相似簇：**494 个**；
- 相似簇内帧：**2,930 张**；
- 删除帧：**2,436 张**；
- 保留帧：**6,277 张**；
- 帧删除率：**27.96%**；
- 保留 crop：**18,386 个**；
- 删除 crop：**7,375 个**。

去重后又使用更宽松阈值分析潜在残余相似帧：

- 场景数：20；
- 帧数：6,277；
- crop 数：18,386；
- 检查帧对：**2,204,461 对**；
- 相似帧对：535 对；
- 相似 crop 对：564 对；
- 相似连通簇：292 个；
- 若每簇仅保留一帧，还可移除 517 帧，约占 8.24%。

这说明我不仅执行了去重，还对去重强度和残余相似度进行了二次审计。

### 3.11 构建防止训练测试泄漏的分组划分策略

#### 问题

视频连续帧或同场景近重复帧如果被随机分到训练集和测试集，会造成数据泄漏，使评测结果虚高。

#### 解决方案

我基于宽松相似帧图的连通分量进行 grouped split：

- 同一个相似帧连通分量只能整体进入训练集或测试集；
- 无相似关系的帧作为单独 group；
- 以 group 为单位逼近 10% 测试比例；
- 输出 group manifest，便于追踪每个样本的划分原因；
- 验证跨 split 相似帧对数量必须为 0。

#### 划分结果

- 总帧：6,277；
- 训练帧 / 测试帧：5,652 / 625；
- 总单区域样本：18,386；
- 训练样本 / 测试样本：16,547 / 1,839；
- 实际测试集比例：10.002%；
- group 数：5,760；
- 相似簇 group：292；
- singleton group：5,468；
- 最大 group：18 帧、37 个 region 样本；
- 跨训练测试集相似帧对：**0**。

该工作显著提高了后续单区域模型评测的可信度。

### 3.12 微调高质量单区域人物描述模型

#### 背景

原始 Florence checkpoint 的人物描述通常较短，对头发、眼镜、鞋子、携带物等属性覆盖不足。Qwen 清洗后的目标描述平均约 22.68 个 token，而基础模型输出平均只有约 16.05 个 token。

#### 训练方案

- 数据：dHash 去重后的 Qwen 人物区域描述；
- 划分：相似簇 grouped split；
- 训练样本：16,547；
- 测试样本：1,839；
- 任务：原生 `<REGION_TO_DESCRIPTION>`；
- 训练配置：1 epoch、batch size 24、learning rate 4e-6；
- 目的：提高人物属性覆盖，同时保持 Florence 原生单区域交互协议。

#### 模型效果

在 1,839 个独立测试区域上，微调模型相对基础 checkpoint 的结果如下：

| 指标 | 基础模型 | 微调模型 | 绝对提升 |
|---|---:|---:|---:|
| Character Similarity | 0.5277 | **0.7347** | +0.2071 |
| Token Precision | 0.6166 | **0.7401** | +0.1235 |
| Token Recall | 0.4530 | **0.7273** | +0.2743 |
| Token F1 | 0.5125 | **0.7252** | +0.2128 |
| ROUGE-L F1 | 0.4594 | **0.6838** | +0.2244 |
| 平均输出 token | 16.05 | **22.05** | +6.00 |

微调后平均输出长度 22.05 token，与参考描述 22.68 token 基本一致，说明模型学会了更完整但不过度冗长的人物描述风格。

#### 人物属性覆盖提升

| 属性 | 基础模型提及率 | 微调模型提及率 | 参考描述提及率 |
|---|---:|---:|---:|
| 头发 | 2.18% | **82.33%** | 77.60% |
| 眼镜 | 1.90% | **14.08%** | 15.12% |
| 上装 | 95.27% | **97.39%** | 96.25% |
| 下装 | 75.04% | **79.77%** | 77.00% |
| 鞋子 | 4.51% | **30.18%** | 31.81% |
| 颜色 | 98.97% | **100.00%** | 100.00% |
| 携带物 | 27.35% | **44.97%** | 42.74% |
| 姿态/动作 | 81.78% | **84.50%** | 79.66% |

其中最明显的改善包括：

- 头发属性提及率从 2.18% 提升到 82.33%；
- 鞋子属性提及率从 4.51% 提升到 30.18%；
- 携带物提及率从 27.35% 提升到 44.97%；
- 多数属性的提及率已接近参考描述分布。

### 3.13 建立人物属性统计与质量分析工具

为评估描述是否包含业务关心的人物属性，我开发了人物属性统计脚本和测试，覆盖：

- 性别与年龄；
- 头发颜色、长度和秃头；
- 眼镜、口罩等面部附件；
- 上装、下装、连衣裙等服装类型；
- 服装颜色和袖长；
- 鞋子；
- 手持、背负、怀抱物品；
- 姿态与动作。

脚本处理了多种容易误判的语言现象，例如：

- `black dress` 同时属于颜色和服装类型；
- 背景颜色不能计入人物服装颜色；
- `holding a backpack` 和 `wearing a backpack` 需要区分手持与背负；
- 手中拿着夹克不能误判为穿着夹克；
- 多个头发细节仍只计作一个头发属性。

项目中为该解析器编写了 19 个针对性测试用例。

### 3.14 开发推理、可视化和模型交付工具

#### CLI 与批量推理

我提供了多种可直接运行的推理工具：

- 单张图片 full caption；
- 单 bbox region description；
- 多 bbox 单次 region descriptions；
- 根据 crop JSON 自动选择单区域或多区域任务；
- 对训练 JSONL 批量推理并统计生成 token 分布；
- 支持 fp16、bf16、设备选择、beam 数和最大输出长度；
- 输出结构化 JSON，记录 prompt、bbox pixel、bbox loc、raw text 和 cleaned text。

#### Web Demo

项目包含两个 FastAPI Web Demo：

1. 多区域 Web Demo：支持图片上传、多个 bbox 输入、Plan A / Plan B 模型选择和结果展示；
2. 单区域 Web Demo：支持反复选择 bbox 并调用最新单区域 checkpoint 推理。

Demo 中包含：

- bbox 合法性检查；
- pixel bbox 到 loc token 转换；
- 模型线程锁；
- checkpoint 和精度配置；
- 自定义 Florence-2 remote code 加载兼容；
- 前端图片、框选和描述结果展示。

#### 模型交付

我整理了独立的 `ugipc_1231_15words_epoch3_full_handoff` 交付包，包括：

- 完整 checkpoint；
- 自定义 Florence-2 modeling、processing 和 configuration 文件；
- tokenizer、special tokens 和 vocab；
- 推理脚本；
- requirements；
- checkpoint manifest；
- 完整 README。

README 明确说明了：

- full image caption 与 region description 的正确 prompt；
- bbox 从 pixel 坐标到 loc token 的量化方式；
- 带 `person` region name 的使用条件；
- 推荐生成参数；
- JSON 输出格式；
- `trust_remote_code=True`、`use_fast=False` 等加载要求；
- 常见错误和交付检查清单。

这使模型产物可以脱离原实验目录进行复现和业务集成。

## 四、主要成果汇总

### 4.1 数据成果

- 完成 3,692 张候选人物图片的数据处理；
- 基于 YOLO26m 建立人物检测和 crop 数据；
- 生成 12,012 个人物区域 Florence 描述，区域推理失败数为 0；
- 构建 8,713 帧、25,761 个人物区域的 Qwen 清洗描述数据；
- 通过 dHash 联合去重保留 6,277 帧、18,386 个区域；
- 删除 2,436 张近重复帧和 7,375 个区域，帧去重率 27.96%；
- 建立 16,547 / 1,839 的无相似帧跨集泄漏单区域训练测试集；
- 建立 2,983 / 331 的初期 Florence 多区域训练测试集。

### 4.2 模型成果

- 为 Florence-2 新增多区域人物描述任务；
- 设计并实现多种 bbox-description 输出协议；
- 完成 16 组主要 checkpoint 实验；
- 多区域推理总体端到端加速 1.78 倍；
- 5 crop 场景端到端加速 2.23 倍；
- Left-to-Right 排序将区域错配率从 19.68% 降至 7.03%；
- 单区域微调模型 Token F1 从 0.5125 提升到 0.7252；
- 单区域微调模型 ROUGE-L F1 从 0.4594 提升到 0.6838；
- 人物头发、鞋子、携带物等属性覆盖率大幅提升。

### 4.3 工程成果

- 建立人物筛选、检测、描述、清洗、去重、训练、评测的一体化流水线；
- 实现多组一键训练评测 Shell 脚本；
- 实现多区域描述匹配与区域错配专项评测；
- 实现速度基准、输出长度和属性分布分析；
- 实现重复样本、错配样本和 Bad Case 可视化；
- 实现单区域和多区域 FastAPI Web Demo；
- 完成可独立交付的模型目录、推理脚本和使用文档；
- 两个目录中的项目脚本、评测与测试代码合计约 **11,448 行**；
- 当前项目包含 **32 个自动化单元测试用例**，覆盖数据构造、bbox 解析、推理任务选择、速度报告和属性解析。

## 五、关键技术难点与解决思路

### 5.1 多区域描述如何与 bbox 正确对应

难点在于生成模型天然输出一段序列，并不保证每条描述严格遵守输入 bbox 顺序。

我的解决思路是：

- 从隐式顺序对应升级为显式 `description + loc` 绑定；
- 对 loc token 进行严格四元组解析；
- 使用最优分配评测定位描述串位；
- 固定人物空间排序；
- 将数量不一致作为强异常并设计 fallback。

### 5.2 如何判断模型是“描述错了”还是“描述给错人了”

普通文本指标无法区分两者。我通过描述与同图所有 GT region 的匹配矩阵，将错误划分为：

- 当前区域匹配；
- 匹配存在歧义；
- 更匹配其他 bbox，即区域错配；
- 与所有 bbox 都不匹配，即内容错误。

这种评测方式让后续改进可以有针对性地处理 grounding，而不是盲目追求平均文本相似度。

### 5.3 如何降低视频连续帧造成的数据泄漏

我没有直接进行普通随机划分，而是先构建相似帧图，将连通分量作为最小划分单位，最终保证跨训练测试集相似帧对为 0。

### 5.4 如何平衡多区域推理速度和生成稳定性

多区域推理减少了视觉编码次数，但输出序列随人物数量增长，可能产生截断、漏描述和后部质量下降。

我通过以下方式平衡：

- 分 crop 数量独立评测速度和准确率；
- 限制单次最大人物数；
- 针对 2～5 crop 统计加速收益；
- 监控 `max_new_tokens` 是否命中；
- 设计数量校验和逐区域 fallback；
- 对高 crop 数场景使用显式 bbox 回显增强鲁棒性。

## 六、实习期间体现的能力

### 6.1 多模态模型应用与微调能力

- 理解 Florence-2 的 vision encoder、文本生成和 loc token 表达方式；
- 能够修改 processor 扩展自定义视觉语言任务；
- 能够完成训练数据协议设计、模型微调、生成后处理和 checkpoint 管理；
- 能够分析多模态生成中的 grounding、错配和灾难性遗忘问题。

### 6.2 数据工程与数据质量治理能力

- 能够构建从原图到结构化训练样本的完整数据生产链路；
- 能够处理批量数据、断点续跑、分片执行、异常记录和结果汇总；
- 能够结合文本、图像和 bbox 信息进行多模态去重；
- 能够识别训练测试泄漏并设计 grouped split。

### 6.3 实验设计与问题分析能力

- 通过受控变量对比输出格式、crop 顺序、数据版本和超参数；
- 不只关注平均指标，还关注按人物数分组和 Bad Case 分布；
- 能够将错误拆分为数量错误、loc 错误、内容错误和区域归属错误；
- 能够根据实验结果形成可执行的下一步改进方案。

### 6.4 工程化与交付能力

- 编写 CLI、批处理脚本、自动训练评测脚本和 Web 服务；
- 建立可复现的输入输出协议与日志字段；
- 编写单元测试验证关键边界条件；
- 整理完整 checkpoint 依赖和交付文档，降低模型接入成本。

## 七、可用于述职或答辩的个人工作表述

在本次实习中，我负责 Florence-2 人物区域描述方向的数据与模型研发。首先，我搭建了从人物图片筛选、YOLO 检测裁剪、区域描述生成、Qwen 描述清洗，到重复数据分析和 dHash 去重的完整数据流水线，最终从 8,713 帧、25,761 个人物区域中构建出 6,277 帧、18,386 个高质量去重区域，并通过相似帧连通分量进行训练测试集分组，保证跨集合相似帧对为 0。

在模型侧，我针对 Florence-2 单 bbox 重复推理效率低的问题，设计并实现了 `<REGIONS_TO_DESCRIPTIONS>` 多区域任务，使模型可以在一次视觉编码后同时描述多个 bbox。我对描述分隔、bbox 显式回显、prompt 设计和 crop 排序进行了多组实验，并建立了能够区分内容错误与区域错配的专项评测体系。最终，多区域方案在 8,324 张图片、23,427 个 crop 上实现 1.78 倍总体端到端加速，在 5 人场景实现 2.23 倍加速；采用从左到右排序后，Wrong BBox Region Rate 从随机顺序的 19.68% 降至 7.03%。

此外，我使用去重后的 Qwen 高质量人物描述数据微调 Florence-2 单区域模型，在 1,839 个测试区域上将 Token F1 从 0.5125 提升至 0.7252、ROUGE-L F1 从 0.4594 提升至 0.6838，并显著提升头发、鞋子和携带物等人物属性覆盖率。最后，我完成了批量推理、自动训练评测、Bad Case 可视化、FastAPI Web Demo 和模型 checkpoint 交付文档，使实验成果具备复现、展示和业务集成能力。

## 八、可用于简历的精炼版本

### 版本一：三条项目经历

- 搭建 Florence-2 人物区域描述数据流水线，完成 Qwen 完整人物筛选、YOLO26m 检测裁剪、区域描述清洗及文本+bbox+dHash 联合去重；从 8,713 帧、25,761 个区域中保留 6,277 帧、18,386 个高质量区域，并通过相似簇分组实现训练测试集近重复样本零交叉。
- 设计并实现 Florence-2 `<REGIONS_TO_DESCRIPTIONS>` 多 bbox 单次推理任务，完成 processor、训练数据协议、模型微调、bbox 显式回显和后处理；在 8,324 张图片、23,427 个区域上取得 1.78× 总体端到端加速，5 人场景加速达到 2.23×。
- 建立多区域描述与 bbox 归属专项评测体系，通过 crop 排序实验将 Wrong BBox Region Rate 从 19.68% 降至 7.03%；使用去重 Qwen 数据微调单区域模型，将 Token F1 从 0.5125 提升至 0.7252、ROUGE-L F1 从 0.4594 提升至 0.6838。

### 版本二：突出算法研究

- 针对 Florence-2 多人物逐框推理效率低和描述串位问题，提出多区域联合生成任务及 `description + loc` 显式绑定协议，系统对比随机、面积、左右和上下排序，验证从左到右排序可将区域错配率降低 12.65 个百分点。
- 设计多区域 grounding 评测方法，基于描述-region Word F1 矩阵与最优分配，将错误细分为 confident match、ambiguous match、wrong bbox assignment 和 content mismatch，支持按帧、按区域及按人物数量进行诊断。
- 构建文本相似度、bbox IoU 与图像 dHash 融合的近重复帧聚类算法，完成代表帧选择、去重数据集生成和连通分量 grouped split，避免视频连续帧造成的测试集泄漏。

### 版本三：突出工程落地

- 完成 Florence-2 人物描述项目从数据生产、模型训练、批量推理、质量评测到 Web Demo 和 checkpoint 交付的端到端工程化，沉淀约 1.1 万行脚本与评测代码、32 个单元测试及多套一键训练评测流程。

## 九、目录与关键产物索引

### 数据工程

- `florence-data/scripts/check_complete_person_qwen.py`
- `florence-data/scripts/crop_qwen_persons_yolo26m.py`
- `florence-data/scripts/clean_region_descriptions_qwen.py`
- `florence-data/scripts/dedup_region_descriptions_exact.py`
- `florence-data/scripts/analyze_scene_frame_crop_similarity.py`
- `florence-data/scripts/build_dhash_dedup_dataset.py`
- `florence-data/outputs/datasets/`
- `florence-data/outputs/dedup/`
- `florence-data/outputs/similarity_analysis/`

### 多区域任务

- `florence-caption/multi_region_description/PLAN.md`
- `florence-caption/multi_region_description/BADCASE_ANALYSIS_AND_IMPROVEMENTS.md`
- `florence-caption/multi_region_description/scripts/prepare_data.py`
- `florence-caption/scripts/train_multi_region.py`
- `florence-caption/scripts/infer_multi_region.py`
- `florence-caption/multi_region_description/scripts/infer_descriptions_from_crops_json.py`

### 评测与实验报告

- `florence-caption/multi_region_description/eval/scripts/eval_plan_b.py`
- `florence-caption/multi_region_description/eval/scripts/evaluate_description_matching.py`
- `florence-caption/multi_region_description/eval/scripts/benchmark_crop_count_sequential.py`
- `florence-caption/multi_region_description/eval/results/crop_order_experiments_comparison.md`
- `florence-caption/multi_region_description/eval/results/crop_count_speed_robustness_tradeoff.md`
- `florence-caption/multi_region_description/eval/results/description_mismatch_loc_token_consistency.md`

### 单区域优化

- `florence-caption/multi_region_description/scripts/prepare_single_region_grouped_split.py`
- `florence-caption/multi_region_description/scripts/train_eval_single_region_dhash10_gpu1.sh`
- `florence-caption/multi_region_description/eval/scripts/eval_single_region_description.py`
- `florence-caption/multi_region_description/checkpoints/qwen_single_region_dhash10_grouped_gpu1_ep1_bs24_lr4e6/`

### 展示与交付

- `florence-caption/multi_region_description/web_demo/`
- `florence-caption/multi_region_description/single_region_web_demo/`
- `florence-caption/ugipc_1231_15words_epoch3_full_handoff/`

## 十、总结

本次实习工作的主要价值在于，将 Florence-2 人物区域描述从一个单点模型能力扩展成了一套较完整的可实验、可评测、可复现、可交付系统。

我不仅完成了数据生产和模型训练，还围绕真实业务中的关键问题进行了系统解决：

- 用多区域单次推理降低多人场景计算成本；
- 用显式 bbox 绑定和固定空间顺序降低描述串位；
- 用多维评测区分内容错误与区域归属错误；
- 用多模态近重复检测提升数据质量；
- 用相似簇分组消除训练测试泄漏；
- 用高质量 Qwen 描述监督提升人物细粒度属性覆盖；
- 用自动化脚本、测试、Demo 和交付文档推动模型成果落地。

最终形成的数据集、模型 checkpoint、评测体系、实验结论和工程工具，可以继续支持人物检索、视频监控理解、ReID 辅助描述、人物属性提取以及多目标视觉语言理解等后续工作。
