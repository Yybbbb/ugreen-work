# Florence-2 人物属性描述模型：SFT + 强化学习全流程实习项目报告

> 项目路径：`/data1/work/MichaelYu/florence-attibute`
> 报告日期：2026-08-12
> 任务类型：区域级人物属性自然语言描述（Region-level Person Attribute Captioning）
> 基座模型：Florence-2-base（231,414,016 参数，全参数微调）

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [技术路线总览](#2-技术路线总览)
3. [模型与任务设计](#3-模型与任务设计)
4. [数据工程](#4-数据工程)
5. [SFT 监督微调实验](#5-sft-监督微调实验)
6. [强化学习阶段（SCST）](#6-强化学习阶段scst)
7. [评估体系](#7-评估体系)
8. [最终结果与结论](#8-最终结果与结论)
9. [工程实现与代码结构](#9-工程实现与代码结构)
10. [踩坑与问题复盘](#10-踩坑与问题复盘)
11. [个人产出与收获](#11-个人产出与收获)
12. [附录](#12-附录)

---

## 1. 项目背景与目标

### 1.1 业务背景

项目服务于**跨镜头人物检索（Cross-camera Person Re-identification / Person Retrieval）**场景。传统 ReID 依赖视觉特征向量做相似度检索，缺点是不可解释、无法用自然语言查询。业务侧希望能支持"用一句话描述找人"的检索方式，例如：

```
查询：一个戴眼镜、背黑色背包、穿白色外套的成年男性
```

要实现这类检索，需要一个能把**监控画面中的单个人物区域**转成**结构稳定、属性密集的自然语言描述**的模型。这个描述后续可以被向量化用于检索，也可以被大模型解析成结构化属性做过滤。

### 1.2 为什么选 Florence-2-base

| 候选方案 | 问题 |
|---|---|
| 大型 VLM（Qwen-VL、InternVL 等） | 参数量大、推理慢，监控场景需要处理海量帧，成本不可接受 |
| CLIP + 属性分类头 | 只能输出固定属性表，无法发现开放属性，不产出自然语言 |
| **Florence-2-base（选用）** | 231M 参数、原生支持区域级 prompt（`<REGION_TO_CATEGORY>` 等）、单卡可部署、有区域定位能力 |

Florence-2 的关键优势是**原生带区域输入接口**：可以直接把人物 bbox 以 `<loc_x>` token 形式喂给模型，不需要裁剪图片、不需要新增任务头。

### 1.3 具体目标

模型输出的 caption 需要同时满足 5 条约束：

1. **只描述人物可见属性**，不描述场景与背景；
2. **理想长度 18–24 个英文单词**；
3. **人物主体位于句首**，整体为单句、句式稳定；
4. **保留开放属性发现能力**，不限制为固定属性表（能说出徽章、手表、图案等 GT 未覆盖的属性）；
5. 尽量**保留 Florence-2-base 的通用能力**（caption、OD、region description）。

目标输出示例：

```text
An adult female with long black hair wears a white jacket, black pants, and glasses while carrying a backpack.
```

明确**不要**的输出形式：JSON、管道分隔字段、严格槽位模板。原因是模板化输出会丢失开放属性发现能力，也无法直接用于语义检索。

---

## 2. 技术路线总览

整个项目分为四个大阶段，历时约三周（2026-07-20 → 2026-08-12）：

```
阶段一：数据构建
  原始 Qwen 标注（21,260 frame / 70,833 crop）
    → 测试集隔离 + 感知哈希去重 + 每帧限流
    → 固定候选池（18,854 frame / 53,602 crop）
    → 场景配额分层采样
    → 人工审核（reviewed）合并回流
    → 五个不可变 split：train 30,000 / dev 2,000 / test 4,328 / rl 5,981 / reserve 21,602

阶段二：SFT 监督微调（5 个版本，V1 → V4B）
  Florence-2-base 全参数微调
    → 隔离变量：global batch (64→32→16)、曝光轮数 (1→1.5→3)、学习率、replay 开关
    → 主指标：Qwen-GT 属性 Micro F1
    → 结论：V4B（batch 16 / 3 epoch / LR×1.25 / 无 replay）最优

阶段三：强化学习 SCST（2 条奖励路线）
  从 V4B final 初始化
    → 混合损失 L = (1-α)·L_CE + α·L_SCST，α: 0.10 → 0.50
    → reward = 0.40·软F1 − 0.40·幻觉率 + 0.10·句式 + 0.10·长度 − 0.10·背景
    → 路线 A（qwen）：Qwen3.5-4B 五档语义裁判做相似度
    → 路线 B（lexical）：token 重叠词法相似度

阶段四：冻结测试集三路对比评估
  4,328 条 test，greedy 解码，统一抽取器 + 统一裁判
    → session 级 2,000 次配对 bootstrap，95% CI
    → 预注册门槛判定 → 推荐 checkpoint
```

### 2.1 关键设计原则

项目全程坚持四条方法论纪律，这是整个实验可信度的基础：

| 原则 | 具体做法 |
|---|---|
| **测试集全程冻结** | test 在任何训练、dev 调参、reward 阈值选择之前锁定；test 只在 checkpoint 确定后跑一次 |
| **单变量隔离** | V1→V2 只改 global batch；V2→V3 只改曝光轮数；V4A→V4B 只改 replay 开关 |
| **预注册门槛** | 推荐门槛（joint F1 提升 > 0.5pp 且 CI 下界 > 0）在评估前写死，不允许事后放宽 |
| **全链路可复现** | 所有数据文件、prompt、reward 代码记 SHA-256；随机种子固定 `20260720`；checkpoint 保存完整 RNG 状态 |

---

## 3. 模型与任务设计

### 3.1 基座权重

从 ModelScope 官方镜像下载：

```text
模型 ID：AI-ModelScope/Florence-2-base
本地路径：pretrained/Florence-2-base（约 888 MB，只读原始副本）
```

离线校验结果：

| 检查项 | 结果 |
|---|---|
| Processor | `Florence2Processor` |
| Tokenizer | `BartTokenizerFast` |
| 模型类 | `Florence2ForConditionalGeneration` |
| 参数量 | 231,414,016 |
| 权重 dtype | FP16（训练时转 BF16 AMP） |

关键文件 SHA-256（防止训练中途权重被改动）：

```text
config.json               c666d0fe0172d46e115e8fba6cd93cd83714575b33a73005cab8d24ce2a3aa8f
model.safetensors         03075d2d2d2bbd3e180b9ba0afae4aa8563226e2d32911656966e05b2f2ee060
processing_florence2.py   f146023a507c009f425a49ee39aa037f4f25c64e14336e3e4f3f1d7377a68e98
```

参数分组（训练时分层学习率的依据）：

| 参数组 | tensor 数 | 参数量 |
|---|---:|---:|
| vision（DaViT 视觉编码器） | 400 | 90,368,000 |
| projection（视觉语言投影层） | 5 | 839,168 |
| language（BART encoder-decoder） | 259 | 140,206,848 |

### 3.2 复用原生任务而非新增任务

这是项目里一个重要的设计决策。人物样本统一使用 Florence-2-base **已有的原生任务**：

```text
<REGION_TO_CATEGORY><loc_x1><loc_y1><loc_x2><loc_y2>
```

下载的 checkpoint 的 `processing_florence2.py` 已包含该任务注册：

```python
self.tasks_answer_post_processing_type["<REGION_TO_CATEGORY>"] = "pure_text"
self.task_prompts_with_input["<REGION_TO_CATEGORY>"] = "What is the region {input}?"
```

因此完全**不需要**：新增任务名、调用 `tokenizer.add_special_tokens()`、调用 `model.resize_token_embeddings()`、修改 task prompt。

**为什么不新增任务？** 新增 special token 会导致 embedding 矩阵扩容，新 token 的 embedding 从随机初始化开始，在 3 万条数据上很难训练充分，还会破坏原有 prompt 分布。复用原生任务则可以直接继承 Florence-2 已有的区域定位与视觉语义能力。

**代价**（明确接受）：`<REGION_TO_CATEGORY>` 在非人物区域上的原始短类别能力会退化。这属于主动的任务重定义，单独测量但不作为硬性回退门槛。

### 3.3 模型实际接收的输入

调用侧字符串：

```text
<REGION_TO_CATEGORY><loc_416><loc_239><loc_510><loc_523>
```

Processor 的 `_construct_prompts()` 展开后，模型实际看到：

```text
What is the region <loc_416><loc_239><loc_510><loc_523>?
```

注意：**输入 prompt 中不添加任何长度要求、背景限制或属性列表**。"18–24 词""不描述背景""人物优先"这些约束全部通过 SFT target 分布、评测门槛和 RL reward 来学习，不靠 prompt 硬编码。这样部署时 prompt 极简，也避免了 prompt 与训练分布错配。

### 3.4 bbox 量化

给定原图宽高 `(W, H)` 和像素 bbox `[x1, y1, x2, y2]`，使用 Florence-2 的 1000-bin floor 量化：

```python
loc_x = floor(x / (W / 1000))
loc_y = floor(y / (H / 1000))
loc   = clamp(loc, 0, 999)
```

**踩坑点**：不能直接复用其他模型标注里已有的 0–999 bbox 字段，必须从原始像素坐标重新量化。不同工具的量化方式（round vs floor、是否归一化到 1000 还是 1024）不一致，直接复用会造成 bbox 系统性偏移。

### 3.5 单区域输入约束

首版每条训练样本只描述**一个**人物 bbox。同一帧中的 8 个人分别形成 8 条独立样本：

```text
full image + person bbox 1 -> caption 1
full image + person bbox 2 -> caption 2
...
```

不使用 `<REGIONS_TO_DESCRIPTIONS>` 风格的多区域联合输出，原因有两点：多区域需要额外设计"区域↔caption"的对齐格式；属性级 reward 的归因难度会显著上升（无法判断某个错误属性是哪个人的）。

输入图像使用**完整 frame**（`768×768`），而不是 person crop。crop 只用于去重、质量计算。这样模型可以利用人物周边的上下文信息（如遮挡关系、相对尺度）。

---

## 4. 数据工程

数据工程占了整个项目约 40% 的工作量，也是最容易出问题的环节。

### 4.1 数据来源

| 用途 | 路径 | 规模 |
|---|---|---|
| 人物 Qwen 标注源 | `/nfs/.../qwen_a35_unified_20260720` | 21,260 frame JSON / ~70,833 person crop |
| 独立测试集 | `/nfs/.../Person_test/pending_review` | 1,945 frame / 4,328 person crop |
| 人工审核结果 | `/data0/.../person_reviewed/frames` | 见 §4.5 |

原始标注由 Qwen 大模型对每个 person crop 生成英文自然 caption + 结构化 attributes。两个 NFS 目录**全程只读**，筛选前后校验文件指纹一致，确认没有污染源数据。

### 4.2 过滤与去重管线

去重严格按顺序执行，先做测试集隔离再做场景内去重：

**第一步：测试集精确隔离**
按 `scene/frame JSON` 相对路径剔除测试集的 1,945 个 frame。

**第二步：跨测试集相似度剔除**
对每个 person crop 计算 64-bit dHash（感知哈希）和 256 维 word/bigram 哈希文本向量：
- dHash 汉明距离 ≤ 8 → 直接剔除；
- dHash 距离 ≤ 16 **且** caption cosine ≥ 0.88 → 剔除。

关键设计：**caption 相似度不能单独触发删除**。因为穿常见颜色服装的不同人物 caption 高度相似（"An adult male wearing a black jacket and black pants"），单靠文本会误删大量合法样本。

**第三步：场景内去重**
只在同一 scene 内继续去重（不做跨 scene 聚类，避免合并外观相似但身份不同的人）：
- dHash 距离 ≤ 10 → 重复；
- dHash 距离 ≤ 18 **且** caption cosine ≥ 0.85 → 重复。

**第四步：每帧限流**
每帧最多保留 8 人，超限时优先保留清晰、无遮挡、截断少、检测置信度高、crop 面积大的样本。

过滤损失明细：

| 项目 | 数量 |
|---|---:|
| 原始 frame JSON | 21,260 |
| 原始 person crop | ~70,833 |
| 精确剔除测试 frame | −1,945 |
| 剔除测试相似 crop | −747 |
| 剔除场景内重复 crop | −7,973 |
| 每帧最多 8 人剔除 crop | −4,166 |
| 源文件权限不足未进入 | −10 个 JSON |
| caption/标注状态无效 | −4 个 crop |
| **最终候选 frame** | **18,854** |
| **最终候选 person crop** | **53,602** |
| 有剩余数据的场景 | 18 |

### 4.3 场景不均衡问题

候选池分布严重长尾：`tradeshow` 有 29,262 人，`company_surveillance_mp4_adaptive` 有 8,880 人，而 `2026_3_30_shopping_mall` 只有几十人。如果按自然比例采样，模型会被两个超大场景主导。

采样策略：

- `tradeshow` 和 `company_surveillance_mp4_adaptive` **各取 7,500 人，各占 SFT 的 25%**（硬上限）；
- 其他 16 个场景合计 15,000 人，按 `sqrt(eligible_scene_people)` 加权分配（平方根压缩长尾，同时保证小场景不被完全淹没）；
- tiny 尺度 crop 不超过 25%；
- 属性丰富度加权：眼镜、帽子、口罩、头盔、背包、手提包、手机、手持物及开放 `extra` 属性提高优先级；
- 保留 caption 长度与表达多样性，**不只选 18–24 词的样本**（否则会人为窄化分布）；
- 不做有放回采样，不复制稀有样本。

### 4.4 划分隔离

先按**完整 session**划分 dev 和 train，任何 session / frame / person sample 都不跨 split。这比按样本随机划分严格得多：同一 session 内的相邻帧几乎是同一个人的连续画面，按样本划分会造成严重的信息泄漏。

Person test 全部来自独立测试目录，不从训练候选池切 test。

### 4.5 人工审核回流

项目中期引入了人工审核环节，对测试集与部分训练数据做了属性修正。审核结果统计：

| 审核状态 | regular | 小红书数据 |
|---|---:|---:|
| reviewed（通过） | 1,887 | 2,050 |
| rejected（拒绝） | 460 | 3,287 |
| duplicate（重复） | 151 | 3,559 |
| uncertain | 1 | 4 |
| unreviewed | 216 | 297 |

合并动作：

| 动作 | 数量 |
|---|---:|
| regular reviewed → test | 1,887 |
| regular duplicate 从 test 移除 | 151 |
| regular rejected 从 test 移除 | 460 |
| 小红书 reviewed → test | 1,025 |
| 小红书 reviewed → train | 1,025 |
| 移除 train-test 泄漏 | 751 |
| 移除 test 与 reviewed train 冲突 | 2 |

**caption 与属性对齐**：人工只修正了结构化 `attributes`，caption 文本仍是旧的。如果直接用会造成"GT caption 说白衣服、GT 属性说黑衣服"的矛盾监督。解决方案是用 Qwen 按修正后的属性**重写 caption**（`data/scripts/rewrite_review_captions.py`，16 个单测覆盖），保证 caption 与 attributes 一致。

合并后 RL split 从 6,000 减为 5,981 人。**这里做了一个刻意的选择：不补齐这 19 人**。补齐意味着要么复制样本、要么引入新样本，两者都会破坏"RL 数据是 SFT train 的严格子集"这一不变量。19/6000 = 0.32% 的规模差异远小于这个不变量的价值。

### 4.6 最终数据规模

| 数据集 | 人物样本数 | frame 数 | 用途 |
|---|---:|---:|---|
| Person SFT train | 30,000 | 15,422 | 主自然 caption SFT |
| Person dev | 2,000 | 944 | checkpoint 选择、reward 校准 |
| Person test | 4,328 | 2,432 | 冻结自动测试（含 2,912 条 reviewed） |
| Person RL train | 5,981 | 3,368 | SFT train 的高多样性严格子集 |
| Person reserve | 21,602 | 6,599 | 不作监督，其中 1,000 人用于 native replay |
| Native replay | 1,000 | 1,000 | 保持原生区域描述能力（仅 SFT 使用） |

场景分布（train / test）：

| 场景 | train | test |
|---|---:|---:|
| tradeshow | 7,403 | 430 |
| company_surveillance_mp4_adaptive | 7,205 | 172 |
| 2026_3_20_expo | 2,447 | 317 |
| 2026_3_12_europe_city | 2,129 | 145 |
| 2026_4_9_ikea | 2,054 | 154 |
| ReID_pedestrian | 1,930 | 237 |
| 2026_04_03_cat_cafe | 1,760 | 321 |
| 2026_3_17_hardware_store | 1,174 | 189 |
| 小红书 | 1,025 | 1,025 |
| 2026_4_16_transport_hub | 697 | 157 |
| 2026_3_09_industrial_park | 583 | 125 |
| 2026_3_25_urban_village | 535 | 144 |
| fall_scene | 397 | 55 |
| company_surveillance | 257 | 187 |
| 2026_4_8_cat_cafe | 224 | 115 |
| 其他 6 个场景 | 190 | 555 |

### 4.7 原生任务 Replay 设计

为降低全参微调的灾难性遗忘，在 SFT 中混入 1,000 条 Florence-2-base 原生区域描述任务：

| 原生任务 | 数量 | 来源 |
|---|---:|---|
| `<REGION_TO_DESCRIPTION>` | 1,000 | reserve（`tradeshow` / `company_surveillance_mp4_adaptive` 各 500 个不同 frame） |

target 由**冻结的 Florence-2-base** greedy 推理生成（自蒸馏），保留原生输出格式。原始 base 的输出可能是"描述文本 + 四个输入区域 loc token"，也可能在困难小目标上退化为 `person<loc_...>`。replay 保留这种原生格式；loc token 必须合法且与输入 bbox 完全一致，空输出、malformed、越界的样本不进入正式 replay。

**关键设计**：replay **只用** `<REGION_TO_DESCRIPTION>`，不含 `<REGION_TO_CATEGORY>`。因为人物样本用的就是 `<REGION_TO_CATEGORY>`，如果 replay 里也放这个任务的原始短类别 target，会与人物自然 caption target 在**同一个任务上**产生直接冲突（同一 prompt 分布下两种截然不同的 target）。

### 4.8 离线 manifest + 流式加载

**问题**：31,000 条样本，如果提前把 `768×768` pixel tensor 缓存下来，会展开成数百 GB。如果训练循环里反复解析原始 scene JSON，CPU 会成为瓶颈。

**方案**：两级设计。

第一级，`scripts/prepare_person_sft_data.py` 流式扫描原始 JSON，生成轻量 JSONL：

```text
data/prepared/train.jsonl          # 30,000 条
data/prepared/dev.jsonl            # 2,000 条
data/prepared/rl.jsonl             # 5,981 条
data/prepared/test.jsonl           # 4,328 条
data/prepared/native_replay.jsonl  # 1,000 条
```

每行只存训练所需的轻量元数据。单行样本示例：

```json
{
  "schema_version": "florence_person_sft_v1",
  "sample_id": "2026_04_03_cat_cafe/2026-04-03_07-55-31_0107.json#0",
  "split": "train",
  "task": "<REGION_TO_CATEGORY>",
  "image": "/data1/.../2026-04-03_07-55-31_0107.jpg",
  "prompt": "<REGION_TO_CATEGORY><loc_714><loc_145><loc_889><loc_734>",
  "label": "An adult female with dark hair pulled back and wearing glasses is dressed in a short-sleeved yellow t-shirt, long white pants, and white shoes.",
  "bbox_xyxy": [914.198, 104.512, 1139.016, 528.943],
  "bbox_loc_0_999": [714, 145, 889, 734],
  "scene": "2026_04_03_cat_cafe",
  "session": "2026_04_03_cat_cafe/2026-04-03_07-55-31",
  "scale": "medium",
  "attributes": { "age_group": "adult", "gender": "female", "upper_garment": {...}, ... }
}
```

第二级，训练启动时只做一次顺序校验，建立 `(文件路径, 字节 offset, task)` 索引，**不把 JSON 对象保留在内存**。DataLoader worker 根据 sampler 给的索引 `seek` 到 offset，读一行后才打开该样本图像；collator 只对当前 batch 调 processor；optimizer step 完成后释放 batch tensor。

内存上界由 `per_device_batch_size × worker 数 × processor 临时 tensor` 决定，**与 31,000 条数据总量无关**。

其他细节：
- train 用 `DistributedSampler(shuffle=True, seed=20260720)` 每 epoch 打乱并按 rank 分片，不做 Python 全量 row shuffle；
- dev 用不打乱的 sampler，保证评测顺序稳定；
- 每个 train worker 维护自己的懒加载文件句柄，不跨进程共享；
- train `prefetch_factor=1`（默认是 2），每 rank 6 个 worker，避免额外 CPU/共享内存压力；
- dev 固定 `num_workers=0`，避免评测时再创建一组 24 个 persistent worker。

### 4.9 固定属性表

评估与 reward 都基于这张固定属性表（19 个标量字段 + 1 个开放列表）：

```text
age_group, gender,
upper_garment.{type, color, length},
lower_garment.{type, color, length},
shoes.{type, color},
head.{accessories, hairstyle, hair_color, hair_length},
carried_items.{handbag, backpack},
handheld_items.{dangerous_item, mobile_phone},
extra[]
```

**unknown 语义约定**：`unknown`、`none`、`no` 及归一化为空的值统一解释为"该属性不存在"。评估时不计入已知属性正例；RL reward 中若模型生成了具体值，计为 fabrication（幻觉）。

---

## 5. SFT 监督微调实验

### 5.1 训练设置

**前处理流程**：

1. 读取完整 frame，`CLIPImageProcessor` 转为 `768×768`；
2. 人物区域用 `expanded_bbox_xyxy` 量化为 4 个 loc token；
3. `BartTokenizerFast` 编码展开后的自然语言 prompt；
4. SFT target 使用 **Qwen 原始英文自然 caption**（不用 JSON、不用固定模板），`max_length=64` tokenizer tokens，padding token 在 labels 中替换为 `-100` 不参与交叉熵。

**硬件**：4 × RTX 6000 Ada（GPU 3-6），torchrun DDP，BF16 AMP + FP32 optimizer state。

**共同配置**：

| 配置 | V1/V2/V3 | V4A/V4B |
|---|---|---|
| 初始化 | `pretrained/Florence-2-base` | 同左（每版独立初始化，非续训） |
| 训练方式 | 全参数更新（231,414,016） | 同左 |
| 优化器 | AdamW，β=(0.9, 0.999)，eps=1e-8 | 同左 |
| Weight decay | 0.01 | 0.015 |
| Label smoothing | 0.05 | 0.05 |
| 梯度裁剪 | 1.0 | 1.0 |
| Warmup / Scheduler | 5% / cosine，floor=10% | 5% / cosine，floor=3% |
| Vision / Projection / Language LR | 1e-7 / 5e-7 / 1e-6 | 2e-7 / 7.5e-7 / 1.25e-6 |
| Target max length | 64 tokens | 64 tokens |
| 每 rank train workers | 6，prefetch=1 | 同左 |
| 每 rank dev workers | 0 | 0 |
| 随机种子 | 20260720 | 20260720 |

**分层学习率的设计理由**：视觉编码器（90M 参数）在预训练中已学到通用视觉特征，任务重定义主要发生在语言侧，因此 vision LR 设为 language LR 的 1/6～1/10。这个比例在 V4 版本整体放大 1.25 倍时保持不变。

### 5.2 五个版本的变量隔离

| 版本 | Run name | 数据 | Global batch | 有效 epoch | Optimizer steps | 隔离的变量 |
|---|---|---|---:|---:|---:|---|
| V1 | `..._replay1k_b64` | 30k + 1k replay | 64 | 1 | 485 | 基线 |
| V2 | `..._replay1k_b32` | 30k + 1k replay | 32 | 1 | 969 | **仅 global batch** 64→32 |
| V3 | `..._replay1k_b32_e1p5` | 30k + 1k replay | 32 | 1.5 | 1,454 | **仅曝光轮数** 1→1.5 |
| V4A | `..._replay1k_b16_e3_lr125` | 30k + 1k replay | 16 | 3 | 5,814 | batch 16 + 3 epoch + LR×1.25 |
| V4B | `..._person_only_b16_e3_lr125` | 30k + **0** replay | 16 | 3 | 5,625 | **仅 replay 开关** 1000→0 |

**为什么这样设计递进**：V1 用 global batch 64 跑完一个 epoch 只有 485 次 optimizer update。训练日志显示 dev loss 从 step 50 的 2.3729 单调降到 step 450 的 1.6270，**完全没有反弹**，说明模型远未收敛，首要问题是更新次数不足。因此 V2 把 global batch 减半到 32（更新次数约翻倍），先隔离 batch 影响；V3 在此基础上把有效曝光提到 1.5 epoch。

V3 的实现细节：不直接跑整数 2 epoch，而是以 `total_optimizer_steps=1454` 作为 scheduler 总步数与停止条件，使每条样本平均曝光 1.5 次。**这里有个易犯的错误**：如果只支持整数 epochs，用 `epochs=2` 配合未调整的 1,938-step scheduler，cosine 会在中途被截断，最后阶段学习率仍然偏高，训练终点的 checkpoint 质量不可控。

V4 同时改了三个变量（batch 16、3 epoch、LR×1.25），严格来说不是单变量隔离，是在 V2/V3 都显示"更多更新 = 更好"后的一次激进推进。V4A→V4B 才是干净的单变量对比（replay 开关）。

### 5.3 训练曲线

**Dev loss / caption 长度分布随 step 变化**（dev 128 条子集，greedy 解码）：

V1（485 step）：

| step | dev loss | 平均词数 | 18–24 词占比 |
|---:|---:|---:|---:|
| 50 | 2.3729 | 5.00 | 0.0% |
| 100 | 1.8831 | 21.90 | 64.1% |
| 200 | 1.7017 | 22.49 | 59.4% |
| 300 | 1.6499 | 22.14 | 58.6% |
| 450 | 1.6270 | 22.23 | 58.6% |

step 50 的平均词数只有 5.0、单句率 0%，说明此时模型仍在输出原生短类别（如 "person"），任务重定义还没完成。step 100 突变到 21.9 词、单句率 100%、主体开头率 99.2%，**句式结构在 100 step 内就基本学会了**，之后的 385 step 主要在提升属性准确率。这也印证了 §5.7 的结论：模板预热阶段完全没必要。

V2（969 step）：

| step | dev loss | 平均词数 | 18–24 词占比 |
|---:|---:|---:|---:|
| 100 | 1.9749 | 20.75 | 64.8% |
| 300 | 1.6600 | 22.25 | 58.6% |
| 500 | 1.5958 | 21.62 | 60.9% |
| 700 | 1.5695 | 21.64 | 61.7% |
| 900 | 1.5606 | 21.52 | 60.2% |

V3（1,454 step）：

| step | dev loss | 平均词数 | 18–24 词占比 |
|---:|---:|---:|---:|
| 100 | 2.0462 | 20.11 | 56.2% |
| 500 | 1.5845 | 21.29 | 60.2% |
| 900 | 1.5265 | 21.46 | 57.8% |
| 1400 | 1.5048 | 21.66 | 56.2% |

V4A（5,814 step）：

| step | dev loss | 平均词数 | 18–24 词占比 |
|---:|---:|---:|---:|
| 200 | 1.9422 | 21.12 | 59.4% |
| 1000 | 1.5010 | 21.12 | 53.1% |
| 2600 | 1.3985 | 21.98 | 50.8% |
| 4200 | 1.3705 | 21.41 | 51.6% |
| 5800 | 1.3643 | 21.38 | 50.0% |

V4B（5,625 step）：

| step | dev loss | 平均词数 | 18–24 词占比 |
|---:|---:|---:|---:|
| 200 | 1.9290 | 21.74 | 56.2% |
| 1000 | 1.4967 | 21.59 | 54.7% |
| 2600 | 1.3935 | 21.35 | 46.9% |
| 4200 | 1.3682 | 21.24 | 51.6% |
| 5600 | 1.3648 | 21.30 | 47.7% |

**曲线读法**：

1. dev loss 单调下降到最后一次评估，五个版本都没有过拟合反弹（V1 1.6270 → V2 1.5606 → V3 1.5048 → V4A 1.3643 → V4B 1.3648）；
2. 更多 optimizer step 带来单调的 loss 改善，V4 相对 V1 降低 0.26；
3. **18–24 词占比随训练变差**（V1 的 58.6% → V4B 的 47.7%），但平均词数几乎不动（22.2 → 21.3）。说明不是整体长度漂移，而是**长度分布变宽**：模型学会了"属性多就多说、属性少就少说"，代价是落在理想区间的比例下降。这正是后续 RL 阶段保留长度奖励的直接依据。

训练耗时（4 卡）：

| 版本 | 起止 | 耗时 |
|---|---|---|
| V1 | 18:16:28 → 18:39:23 | 22 分 55 秒 |
| V2 | 18:42:38 → 19:05:04 | 22 分 26 秒 |
| V3 | 19:08:30 → 19:49:40 | 41 分 10 秒 |
| V4A | 19:52:35 → 21:05:30 | 1 小时 12 分 55 秒 |
| V4B | 21:08:50 → 22:20:50 | 1 小时 12 分 00 秒 |

五个版本串行跑完约 3 小时 52 分钟。V1 和 V2 耗时几乎相同（同样 1 epoch，只是 optimizer step 数不同，前向反向计算量一致）。

### 5.4 SFT 测试集结果（人工 GT 口径）

GT 为测试样本中的人工 `attributes`，预测侧是 Qwen 从 Florence caption 抽取的属性：

| 版本 | Micro P / R / F1 | Soft Micro F1 | Macro-field F1 | Mean field exact | Unknown-extra | 平均词数 | P95 | 18–24 词 | 背景关键词 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 0.6755 / 0.5522 / 0.6076 | 0.6733 | 0.4743 | 0.5522 | 27.38% | 21.75 | 29 | **59.08%** | 0.116% |
| V2 | 0.6836 / 0.5598 / 0.6155 | 0.6796 | 0.4829 | 0.5598 | 26.70% | 21.51 | 29 | 55.91% | 0.139% |
| V3 | 0.6896 / 0.5647 / 0.6209 | 0.6842 | 0.4888 | 0.5647 | 26.01% | 21.42 | 30 | 55.22% | 0.069% |
| V4A | 0.7056 / 0.5756 / 0.6340 | 0.6947 | 0.5070 | 0.5756 | 24.64% | 21.34 | 30 | 48.59% | 0.069% |
| **V4B** | **0.7075 / 0.5769 / 0.6356** | **0.6962** | **0.5147** | **0.5769** | **24.55%** | 21.32 | 30 | 49.12% | 0.069% |

### 5.5 SFT 测试集结果（Qwen-GT 口径，主指标）

人工 GT 的词汇体系与 Qwen 抽取 schema 存在系统性错配（例如人工写 "dark"、Qwen 抽出 "black"），这会给所有版本叠加一层固定的口径惩罚。为消除这一偏差，另用 Qwen 从 **GT caption** 抽取的属性作为 GT（`data/prepared/test_qwen_attributes.jsonl`）——这样预测侧和 GT 侧经过**同一个抽取器**，是当前版本比较的主指标：

| 版本 | Micro P / R / F1 | Soft Micro F1 | Macro-field F1 | Mean field exact | Unknown-extra | 平均词数 | P95 | 18–24 词 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | 0.6822 / 0.6118 / 0.6451 | 0.7150 | 0.5098 | 0.6118 | 14.79% | 21.75 | 29 | **59.08%** |
| V2 | 0.6906 / 0.6219 / 0.6545 | 0.7228 | 0.5230 | 0.6219 | 13.84% | 21.51 | 29 | 55.91% |
| V3 | 0.6985 / 0.6292 / 0.6620 | 0.7295 | 0.5324 | 0.6292 | 13.03% | 21.42 | 30 | 55.22% |
| V4A | 0.7162 / 0.6464 / 0.6795 | 0.7436 | 0.5573 | 0.6464 | 10.99% | 21.34 | 30 | 48.59% |
| **V4B** | **0.7183 / 0.6477 / 0.6812** | **0.7447** | **0.5647** | **0.6477** | **10.94%** | 21.32 | 30 | 49.12% |

**核心结论**：

1. V1→V2→V3→V4 属性指标**单调提升**，验证了"更多 optimizer update"这条主线判断正确；
2. 相对 V1，V4B 的 Micro F1 **+3.61pp**、Macro-field F1 **+5.49pp**、unknown-extra rate **−3.85pp**；
3. **replay 的边际收益很小**：V4A（1,000 replay）vs V4B（0 replay），主指标 F1 差 0.17pp，V4B 反而略高。说明在 30,000 条人物数据规模下，1,000 条 replay 占比仅 3.2%，不足以显著改变遗忘程度，反而稀释了人物任务的梯度；
4. 长度是主要短板，18–24 词比例从 59.08% 降至 49.12%。

**V4B 选为 RL 起点**，其 Qwen-GT 主指标为：Micro F1 = 0.6812，Soft Micro F1 = 0.7447，Macro-field F1 = 0.5647。

### 5.6 逐字段 positive F1（Qwen-GT 口径）

| 字段 | V1 | V2 | V3 | V4A | V4B |
|---|---:|---:|---:|---:|---:|
| age_group | 0.8356 | 0.8381 | **0.8456** | 0.8427 | 0.8439 |
| gender | 0.8800 | 0.8818 | 0.8842 | **0.8978** | 0.8976 |
| upper_garment.type | 0.4997 | 0.5082 | 0.5166 | 0.5514 | **0.5523** |
| upper_garment.color | 0.5737 | 0.5813 | 0.5857 | 0.6074 | **0.6112** |
| upper_garment.length | 0.7289 | 0.7509 | 0.7623 | 0.7933 | **0.7970** |
| lower_garment.type | 0.7890 | 0.7948 | 0.8060 | **0.8248** | 0.8214 |
| lower_garment.color | 0.5076 | 0.5229 | 0.5461 | 0.5581 | **0.5613** |
| lower_garment.length | 0.7677 | 0.7825 | **0.7927** | 0.7955 | 0.7956 |
| shoes.type | 0.5461 | 0.5558 | 0.5632 | **0.6077** | 0.6051 |
| shoes.color | 0.5426 | 0.5524 | 0.5608 | 0.5821 | **0.5838** |
| head.accessories | 0.3953 | 0.4270 | 0.4334 | 0.4688 | **0.4742** |
| head.hairstyle | 0.2125 | 0.2603 | 0.2770 | 0.3570 | **0.3685** |
| head.hair_color | 0.4767 | 0.4864 | 0.4916 | 0.5150 | **0.5196** |
| head.hair_length | **0.8544** | 0.8538 | 0.8486 | 0.8371 | 0.8351 |
| carried_items.handbag | 0.0577 | 0.0664 | 0.0707 | 0.0871 | **0.0954** |
| carried_items.backpack | 0.2263 | 0.2300 | 0.2488 | 0.3229 | **0.3416** |
| handheld_items.dangerous_item | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0741** |
| handheld_items.mobile_phone | 0.2822 | 0.3216 | 0.3491 | 0.3818 | **0.3867** |

主要增益集中在上衣类型/颜色/长度、下衣类型、鞋、头部配饰、发型、背包、手机。`head.hair_length` 是唯一在 V1 达到最优并随长训练**回退**的字段（0.8544 → 0.8351），推测是长训练让模型倾向于用更具体的 hairstyle 描述替代笼统的 hair_length。危险物字段在 V1–V4A 全为 0，V4B 首次非零（0.0741），但 GT 正例只有 25 条，不具统计意义。

### 5.7 被取消的模板预热对照

原计划中有一个"模板预热"阶段（0.25 epoch 模板 target + 0.75 epoch 自然 caption），触发条件是主 SFT 的单句率 < 98% 或人物主体开头率 < 95%。

**实际未触发**。事前统计显示 Qwen caption 中 **99.94% 是单句、99.96% 使用 wear/wearing 结构**，句式监督信号已经足够强；训练曲线也显示句式在 100 step 内就学会（V1 step 100 单句率 100%、主体开头率 99.2%）。V4B 最终测试集单句率 100%、主体开头率 98.22%，两项都远超门槛。

这是一个"计划中的实验因为前置条件不满足而正确地被跳过"的例子，节省了一次完整训练的成本。

---

## 6. 强化学习阶段（SCST）

### 6.1 为什么用 SCST 而不是 PPO/DPO

| 方案 | 是否采用 | 理由 |
|---|---|---|
| PPO | 否 | 需要额外训练 value network，对 231M 模型是显著的额外成本；超参敏感 |
| DPO | 否 | 需要成对偏好数据，而我们的奖励是可计算的标量（属性 F1），构造偏好对反而丢信息 |
| **SCST（Self-Critical Sequence Training）** | 是 | caption 任务的经典做法；baseline 直接用当前 policy 的 greedy 输出，**不需要额外的 critic 网络**；奖励可直接由属性抽取 + 匹配计算 |

SCST 的核心思想：对每个样本采样一条 caption（`temperature=0.8, top_p=0.95`）和一条 greedy caption，两者的 reward 差值作为 advantage：

```text
advantage = R(sampled) − R(greedy)
L_SCST = −advantage × log P(sampled)
```

greedy 输出作为自身 baseline，天然满足"比自己当前的确定性输出更好就鼓励"。

### 6.2 混合损失

纯 SCST 会让模型为了刷 reward 而退化成属性列表（"male, black jacket, black pants, white shoes"），丢掉自然语言句式。因此用 CE 作为锚点：

```text
L = (1 − α) · L_CE + α · L_SCST
```

`α` 在 748 个 optimizer step 上**从首步 0.10 线性增加到末步 0.50**。`L_CE` 是 5,981 条人物 caption 的 teacher-forced 交叉熵（RL 阶段**不使用** native replay，CE 锚点只来自这批人物 caption）。

设计意图：前期以 CE 为主，保护 SFT 学到的句式与分布；后期逐步放开 SCST 权重，让 reward 有机会真正改变行为。α 上限 0.50 而不是 1.0，保证 CE 永远至少占一半权重。

### 6.3 奖励函数设计

这是 RL 阶段最核心的设计。奖励由**一个主项 + 一个盲区补充 + 三个小项**组成：

```text
R = + 0.40 × F1                  # 属性准确率（主项）
    − 0.40 × fabrication_ratio   # 凭空多说（主项，F1 的盲区）
    + 0.10 × sentence_structure  # 句式
    + 0.10 × length_score        # 长度（带符号）
    − 0.10 × background_penalty  # 背景
```

空输出或非法格式（空串 / JSON 开头 / 管道分隔字段）直接 `R = −1.0`；最终裁剪到 `[−1, 1]`。

#### 6.3.1 逐字段软分类

对固定属性表的每个字段，按 GT 值与生成值分类。双方都是具体值时，由 `similarity_fn(field, gt, gen)` 给出相似度 `s ∈ [0,1]`，按连续化公式计数：

| GT 值 | 生成值 | 计入 |
|---|---|---|
| unknown | unknown | 忽略（不计分） |
| **unknown** | **具体** | `n_fab += 1`（多说，F1 盲区），`n_gen_assert += 1` |
| 具体 | unknown | `soft_fn += 1.0`（少说 / miss） |
| 具体 | 具体 | `soft_tp += s`，`soft_fp += 1−s`，`soft_fn += 1−s`，`n_gen_assert += 1` |

开放 `extra[]` 用贪心匹配（阈值 0.5）：
- 命中 GT extra（最佳相似度 ≥ 0.5）→ `soft_tp += s`，`soft_fp += 1−s`，`soft_fn += 1−s`；
- 未命中任何 GT extra → `n_fab += 1`（虚构 extra），`n_gen_assert += 1`；
- GT 有但生成未覆盖 → `soft_fn += 1.0`（recall miss）。

由此：

```text
F1 = 2·soft_tp / (2·soft_tp + soft_fp + soft_fn)     # 分母为 0 时 F1 = 0
fabrication_ratio = n_fab / max(n_gen_assert, 1)
```

**职责分离（避免双重计数）**：F1 只管"已知属性说对/说错/漏说/覆盖 GT-extra"；fabrication 只管"GT 没有却凭空生成"。一个未匹配的 extra 只进 fabrication，不进 F1 的 fp。这个分离是刻意的——如果让 fabrication 同时进 F1 的 fp，等于对同一个错误惩罚两次，权重就失控了。

**为什么 fabrication 需要单独一项**：GT 经人工审核后已近乎完整，SFT 阶段最严重的问题是"多说不存在的属性"（unknown-extra rate 10.94%）。而 F1 对这类错误是**完全盲的**：GT 字段为 unknown 时不计入任何正负样本，模型编造一个值不会降低 F1。所以必须有一个 F1 看不到的独立惩罚项。给它和 F1 完全相同的权重（各 0.40），表示"说对"和"不乱说"同等重要。

#### 6.3.2 三个格式小项

| 项 | 取值 | 判定 |
|---|---|---|
| `sentence_structure` | 0 / 1 | 单句（`[.!?]` 出现 ≤ 1 次）且人物主体靠近句首；JSON、管道字段、碎片列表得 0 |
| `length_score` | +1.0 / −0.5 / −1.0 | 18–24 词 → +1.0；12–17 或 25–28 词 → −0.5；< 12 或 > 28 词 → −1.0 |
| `background_penalty` | 0 / 1 | caption 出现场景、建筑、位置等非人物背景关键词时为 1.0 |

`length_score` 是**带符号**的（贡献范围 −0.10 ~ +0.10），而不是 0/1 门槛。这样"稍微超出理想区间"和"严重超长"有区分度，模型不会因为一点点越界就完全放弃。

### 6.4 两条奖励路线

`similarity_fn` 可切换，同一套 `classify_sample + compute_reward` 支持两条路线：

| 路线 | similarity_fn | 特点 |
|---|---|---|
| `lexical` | `value_similarity`（token 重叠，与评估器 soft F1 **同款**） | 与评估指标紧对齐；同义词（white / light-colored）判 0，不给奖励 |
| `qwen` | Qwen3.5-4B 五档语义裁判 | 同义/近义给分（white ≈ light-colored ≈ 1.0）；与词法评估 F1 **故意分歧** |

语义裁判的五档 prompt（输出 `{1.0, 0.75, 0.5, 0.25, 0.0}`）：

```text
- 1.00  完全匹配或同义 / 同一值的不同具体度
        "white" vs "light-colored" | "dark" vs "black" | "navy" vs "dark blue"
- 0.75  强相关（同类别，非常接近）
        "jacket" vs "coat" | "pants" vs "jeans"
- 0.50  部分：泛化词弱覆盖具体词
        "top" vs "jacket" | "shoes" vs "sneakers"
- 0.25  弱相关（同大类，明显不同）
        "shirt" vs "jacket" | "blue" vs "purple"
- 0.00  冲突：确实是不同的值，即使属于同类属性
        "blue" vs "red" | "male" vs "female" | "short hair" vs "long hair"
```

解析时把 score snap 到最近的允许档；任何解析失败**保守回落到 0.0**（视为冲突，倾向惩罚而非放过幻觉）。

**为什么要跑两条路线**：这是一个真实的开放问题——评估指标用的是词法匹配，那么训练奖励应该也用词法（紧对齐、优化信号直接可迁移），还是应该用语义（更接近人类判断，但与评估口径分歧）？两者都有道理，所以做成对照实验。事前的零成本验证（128 条 SFT dev 预测，用 35B 模型作抽取+裁判替身，算 Spearman 排序相关性）显示：

| reward | vs hard_F1 | vs soft_F1 |
|---|---:|---:|
| 词法 F1 核心 | +0.68 | +0.45 |
| Qwen 语义 | +0.59 | +0.59 |
| 旧版 split 设计（词法 rule_judge） | +0.54 | +0.75 |

合并 F1 核心的对齐优于旧的 split 设计；词法路线对词法评估 F1 最紧，qwen 路线略低（同义分歧所致）。

### 6.5 Reward 服务架构

奖励管线依赖两个冻结的 Qwen 模型，通过 vLLM 以 OpenAI 兼容接口提供服务（**不使用 CLIP，不使用文本 embedding 模型**）：

| 角色 | 模型 / 地址 | GPU | vLLM 配置 |
|---|---|---:|---|
| 属性抽取 | `Qwen3.6-27B-FP8` / `http://127.0.0.1:6097/v1` | 1 | vLLM 0.19.1，context 65,536，并发 64，单批 token 8,192，`VLLM_TEST_FORCE_FP8_MARLIN=1` |
| 值裁判 | `Qwen3.5-4B` / `http://127.0.0.1:6098/v1` | 2 | context 32,768，并发 128，单批 token 65,536，swap 8 GiB |

两个服务均为**纯文本模式**（只输入 caption，不输入图像）、`enable_thinking=false`、`temperature=0`、`do_sample=False`。

**关键约束**：
- 抽取器按固定属性表输出 JSON，未出现的字段**必须填 `unknown`**（不得省略字段）——省略字段会让下游无法区分"模型没提到"和"抽取器漏了"；
- checkpoint、tokenizer、抽取 prompt、JSON schema 在 RL 开始前固定版本并记 SHA-256；
- 抽取结果按 `caption_sha256 + extractor_model_sha256 + prompt_version` 缓存；裁判结果按 `(field, gt_value, gen_value)` 缓存。缓存是必需的——一次 RL run 要评估 748×8×2 ≈ 12,000 条采样 caption，加上 dev 评估，重复的 (field, gt, gen) 三元组极多；
- `VLLM_TEST_FORCE_FP8_MARLIN=1` 是为了避开 Ada 架构上 block-FP8 Triton kernel 的非法访存问题。

**启动与预检**：

```bash
python3 scripts/run_rl_reward_services.py print-command all
python3 scripts/run_rl_reward_services.py status all
python3 scripts/run_rl_reward_services.py start all
```

`start` 只创建缺失的固定名称容器；若同名容器已存在但异常，命令**拒绝** stop/remove/replace（这是刻意的设计：自动替换容器可能杀掉别人正在用的服务）。

RL 训练在**加载 Florence 权重之前**由 rank 0 执行预检：检查 `/v1/models` 返回的模型名，再发一个确定性 `OK` 请求并要求 `content == "OK"` 且 reasoning 为空（验证 thinking 确实关闭）。结果广播到全部 rank，失败时一致退出。`lexical` 路线只检查抽取器，`qwen` 路线检查两者。`--skip-reward-service-preflight` 只用于诊断，该状态会写入 run invariants。

**为什么预检要在加载权重之前**：Florence 权重加载 + DDP 初始化要几十秒，如果 reward 服务不可用，等到第一个 step 才发现就浪费了这些时间，而且此时已经占了 4 张 GPU 的显存。

### 6.6 RL 训练配置

| 配置 | 数值 |
|---|---:|
| 初始化 | `artifacts/sft/..._person_only_b16_e3_lr125/final`（V4B） |
| RL 数据 | `data/prepared/rl.jsonl`，5,981 条，1 pass |
| Global batch | 8（每卡 2 × 4 卡 × 累积 1） |
| Optimizer steps | 748 |
| 文本 / 投影学习率 | 2e-7 |
| 视觉编码器学习率 | 5e-8 |
| Sampling temperature | 0.8 |
| Top-p | 0.95 |
| Baseline | 当前 policy 的 greedy 输出 |
| α 调度 | 0.10 → 0.50 线性 |
| 精度 | BF16 |
| Weight decay | 0.01 |
| 梯度裁剪 | 1.0 |
| 评测间隔 | 100 optimizer steps |
| NCCL timeout | 60 分钟 |

RL 学习率比 SFT 低一个数量级（2e-7 vs 1.25e-6）：SCST 的梯度方差远大于 CE，且我们只想在 V4B 基础上做保守微调，不想破坏已有能力。

**评测协议**：每 100 step 对全部 2,000 条 dev 计算分布式 teacher-forced CE，并对**固定的前 128 条** dev 运行 greedy caption reward。128 条按原始 index round-robin 分到 4 个 rank，完成后由 rank 0 校验无缺失/重复并恢复原始顺序。属性抽取按每请求 4 条分块；裁判 pair 先去重，再由每个 rank 最多 16 并发预取（全局最多 64，低于服务并发 128）。

checkpoint 选择：按平均 greedy dev reward 降序、dev CE 升序、step 升序取 top-3；训练正常退出时**始终额外保留 `final/`**，不受 top-3 淘汰影响。

**完整可恢复性**：所有 checkpoint 保存 model、processor、optimizer、scheduler、GradScaler、epoch、下一 micro-batch、global step、top-3 元数据及**每个 rank 的 Python/CPU/CUDA RNG 状态**。`--resume-from` 恢复这些状态并跳过已消费的 batch；配置、数据 hash、alpha、生成参数或 reward 服务/模型不一致时**拒绝恢复**。这个严格性在实际运行中救了一次（见 §10.2）。

### 6.7 RL 训练曲线

**路线 A：qwen（语义裁判）**，2026-08-10 21:44 → 2026-08-11 07:12，耗时 9 小时 28 分：

| step | dev loss | mean reward | soft F1 | fabrication | 平均词数 | 18–24 词 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.3636 | 0.3861 | 0.7037 | 4.40% | 21.47 | 50.78% |
| 200 | 1.3624 | 0.3799 | 0.6861 | 4.39% | 21.45 | 51.56% |
| 300 | 1.3618 | 0.3893 | 0.7074 | 3.98% | 21.10 | 50.78% |
| 400 | 1.3618 | 0.3882 | 0.7041 | 4.41% | 21.09 | 52.34% |
| 500 | 1.3623 | 0.3851 | 0.6875 | 3.83% | 20.90 | 53.13% |
| **600** | 1.3632 | **0.3927** | 0.7113 | 4.32% | 20.99 | 53.13% |
| 700 | 1.3635 | 0.3759 | 0.6684 | 4.11% | 20.95 | 53.13% |

**路线 B：lexical（词法）**，2026-08-11 07:12 → 16:36，耗时 9 小时 24 分：

| step | dev loss | mean reward | soft F1 | fabrication | 平均词数 | 18–24 词 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.3631 | 0.3836 | 0.7025 | 4.73% | 21.50 | 50.00% |
| 200 | 1.3642 | 0.3751 | 0.6832 | 5.00% | 21.65 | 50.78% |
| 300 | 1.3622 | 0.3821 | 0.6811 | 4.62% | 21.25 | 54.69% |
| 400 | 1.3630 | 0.3765 | 0.6585 | 4.35% | 21.02 | 56.25% |
| 500 | 1.3638 | 0.3946 | 0.7047 | 4.05% | 20.95 | 55.47% |
| **600** | 1.3661 | **0.3972** | 0.7059 | 4.11% | 20.99 | **57.03%** |
| 700 | 1.3685 | 0.3833 | 0.6879 | 4.42% | 20.90 | 53.91% |

两条路线的单句率与主体开头率**全程 100%**，背景关键词率**全程 0%**，空输出 0 条。

**曲线读法**：

1. **dev reward 提升幅度很小**（qwen: 0.386 → 0.393；lexical: 0.384 → 0.397），且非单调，波动幅度与提升幅度同量级。这是 748 step、global batch 8、学习率 2e-7 的保守配置下的预期行为；
2. **dev CE 基本不动**（1.362 ~ 1.369），说明 CE 锚点起了作用，模型没有为刷 reward 而偏离自然语言分布。lexical 路线后期 CE 略有上升（1.3622 → 1.3685），是 α 增大到 0.5 后 SCST 权重上升的正常表现；
3. **长度约束明确生效**：两条路线的平均词数都从 21.5 降到 20.9，18–24 词占比都上升（lexical 从 50.0% 到 57.0%）。这是奖励函数中 `length_score` 项的直接效果；
4. **fabrication 有下降趋势但不明显**（dev 128 条样本太小，噪声大）；
5. 训练日志中的 `advantage` 多数为负（如 `r_s=0.318 r_g=0.388 advantage=-0.070`），说明采样输出通常**不如** greedy 输出——这在 SCST 中是正常的，采样引入的随机性大多是有害的，梯度在推动模型远离这些差的采样。

耗时分析：9.5 小时里绝大部分花在 reward 计算上（每 step 要对 8 条采样 + 8 条 greedy 共 16 条 caption 调抽取器，qwen 路线还要调裁判）。纯 Florence 前向反向只占很小比例。这也是为什么必须做缓存和并发预取。

---

## 7. 评估体系

评估体系是这个项目里投入最多设计精力的部分。核心难点：**如何客观衡量一段自然语言 caption 的属性准确率**。

### 7.1 评估管线

```
Florence checkpoint
  → greedy 解码（do_sample=False, num_beams=1, max_new_tokens=64）
  → 4,328 条 test caption
  → Qwen3.6-27B-FP8 抽取为 canonical 属性 JSON（temperature=0, thinking disabled）
  → 与 GT 属性逐字段配对
  → 相似度加权 soft F1（两套相似度函数：lexical / Qwen 五档语义）
  → session 级 2,000 次配对 bootstrap → 95% CI
```

**关键点：预测侧和 GT 侧经过同一个抽取器**。这消除了"人工标注词汇 vs 模型输出词汇"的系统性偏差。GT 属性来自 `data/prepared/test_qwen_attributes.jsonl`——用同一个 Qwen 抽取器从**人工修正后的 GT caption** 反向抽取。

抽取的可靠性数据：共提交 4,352 个请求，初始批处理失败 24 条，逐条重试 24 条后**最终失败数为 0**。最终属性文件 4,328 行、无缺失 GT、无重复行、无提取失败。

### 7.2 软 F1 计数定义

逐字段将 GT 与模型抽取属性配对，用相似度加权计数，而非硬性 0/1 判定：

| 计数项 | 含义 |
|---|---|
| `gt_positive` | GT 该字段有明确值（非 unknown）的样本数，即"应该答对"的题量 |
| `n_gen_assert` | 模型给出具体值的次数，即"作答"总量 |
| `soft_tp` | 相似度累加 `Σs`，`s ∈ [0,1]` |
| `soft_fp` | `Σ(1−s)`，作答但不完全正确的部分 |
| `soft_fn` | `Σ(1−s)` + GT 有值而模型未提及的次数 |
| `n_fab` | GT 确为 unknown 但模型作答的次数（幻觉） |

```text
Precision = soft_tp / (soft_tp + soft_fp)
Recall    = soft_tp / (soft_tp + soft_fn)
F1        = 2PR / (P + R)
```

**为什么用 soft 而不是 hard**：hard 匹配下 "navy" vs "dark blue" 判 0 分，这对模型不公平；但完全依赖语义裁判又引入模型不确定性。soft F1 用连续相似度做加权，同时报告两套相似度函数下的结果，让读者自己判断。

### 7.3 核心指标

| 指标 | 定义 | 方向 |
|---|---|---|
| **Lexical Micro P/R/F1** | 相似度用 token 重叠（`value_similarity`）计算，全字段汇总后算 micro 平均。严格、可复现、无模型依赖 | 越高越好 |
| **Qwen Semantic Micro P/R/F1** | 相似度改用 Qwen3.5-4B 五档评分，能识别 "navy" ≈ "dark blue"。更接近人类判断 | 越高越好 |
| **Joint Attribute F1** | `0.5 × (Lexical F1 + Semantic F1)`，**主决策指标** | 越高越好 |
| **Lexical Macro-field F1** | 各字段 F1 的算术平均（不按样本量加权）。低频字段权重被放大，反映"冷门属性"表现 | 越高越好 |
| **Fabrication ratio** | `n_fab / n_gen_assert`。F1 的盲区补充 | 越低越好 |

**Micro vs Macro 的必要性**：Micro 按样本量加权，反映整体体验；Macro 平等对待 19 个字段，能暴露"高频字段涨、低频字段跌"的情况。项目最终结果正好出现了这个分歧（见 §8.4），说明两个指标都报是必要的。

### 7.4 质量门槛指标

| 指标 | 定义 | 门槛 |
|---|---|---|
| **Structure pass rate** | caption 为单句且以主语开头的比例 | 绝对值 ≥ 95%，相对基线不得下降 > 1pp |
| **Background leakage rate** | caption 提及场景/建筑/地点/背景的比例 | 绝对值 ≤ 1% |
| **Invalid output rate** | 空输出、JSON 残留、管道分隔字段等格式崩坏比例 | 相对基线不得上升 > 0.5pp |

### 7.5 参考性指标（不参与判定）

| 指标 | 定义 |
|---|---|
| Length mean | 平均词数 |
| 18–24 ratio | 落在 18–24 词理想区间的比例 |

长度指标**降级为参考项**是项目后期的一个判断修正。原计划把"18–24 词比例 ≥ 85%"作为发布硬门槛，实际发现这不合理：有些人物可见属性本来就多（穿戴复杂、带多件物品），有些本来就少（远景小目标只能看出性别和上衣颜色）。强行压到统一区间会牺牲属性召回。符合区间的样本多当然更好，但**不具决定性**。

### 7.6 统计方法

- **配对差值**：同一批样本、同一套 GT、同一抽取器，三个候选唯一差异是 Florence checkpoint，因此可做严格配对比较；
- **95% 置信区间**：以 **session** 为重采样单位，2,000 次 bootstrap，取差值分布的 2.5 / 97.5 分位数；
- **显著性判据**：CI 完全排除 0 即为统计显著；
- **预注册推荐门槛**：joint F1 提升 **> 0.5pp** 且 CI 下界 > 0。

**为什么按 session 而不是按样本 bootstrap**：同一 session 内的样本高度相关（连续帧、同一批人物、同一光照条件）。按样本重采样会把这些相关样本当成独立观测，**系统性低估方差**，导致 CI 过窄、把噪声判成显著。按 session 重采样把整个 session 作为一个独立单位，是更保守也更正确的做法。

### 7.7 通用能力保持评估（未执行）

原计划在互斥的 COCO 保持集上比较训练模型与 Florence-2-base：caption CIDEr / CLIPScore、detailed caption BERTScore / CLIPScore、OD mAP、region description 相对 base 的 BERTScore、non-person region category 准确率。通用综合指标只由 caption、detailed caption、OD 和 region description 组成，**不包含被主动重定义的 `<REGION_TO_CATEGORY>`**。

**这一项截至报告日期尚未执行**，是发布决策的剩余空缺项。诚实地说，这是本项目未完成的部分——虽然 replay 机制和低学习率设计都是为了控制遗忘，但没有量化验证。RL 阶段的风险相对较低（只跑了 748 step、学习率 2e-7、CE 锚点占一半以上权重），SFT 阶段（5,625 step 全参更新）的遗忘程度是真正的未知项。

### 7.8 小规模人工测试（未执行）

原计划从锁定 test 中分层抽取约 200 条做人工核验，标记：正确属性、可见但遗漏的属性、无视觉依据的属性、背景内容、以及三个候选的盲选偏好。人工结果与自动 Qwen 指标分开报告，不用少量人工样本重新拟合阈值。

这一项也未执行。作为替代，构建了可视化 web demo（§9.4）做定性抽查。

---

## 8. 最终结果与结论

### 8.1 三个候选

| 候选 | Checkpoint |
|---|---|
| `v4b_sft`（基线） | `artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final` |
| `qwen_rl` | `artifacts/rl/region_category_person_scst_reviewed5981_qwen4b/final` |
| `lexical_rl` | `artifacts/rl/region_category_person_scst_reviewed5981_lexical/final` |

测试集 4,328 条，greedy 解码，session 级 2,000 次 bootstrap，seed 20260720。

### 8.2 核心指标对比

| 指标 | v4b_sft（基线） | qwen_rl | lexical_rl | 最优 |
|---|---:|---:|---:|:---:|
| **Joint Attribute F1** | 77.089% | 76.933% | **77.296%** | lexical_rl |
| Lexical Micro F1 | 74.218% | 74.000% | **74.580%** | lexical_rl |
| — Precision | 78.554% | 78.335% | **78.757%** | lexical_rl |
| — Recall | 70.336% | 70.119% | **70.824%** | lexical_rl |
| Qwen Semantic Micro F1 | 79.959% | 79.866% | **80.011%** | lexical_rl |
| — Precision | **84.634%** | 84.542% | 84.488% | v4b_sft |
| — Recall | 75.774% | 75.679% | **75.985%** | lexical_rl |
| Lexical Macro-field F1 | 59.743% | 59.455% | **59.926%** | lexical_rl |
| **Fabrication ratio** ↓ | 11.072% | **10.633%** | 10.806% | qwen_rl |
| Structure pass rate | **98.221%** | **98.221%** | 98.198% | 并列 |
| Background leakage ↓ | 0.069% | 0.069% | 0.069% | 并列 |
| Invalid output ↓ | 0.000% | 0.000% | 0.000% | 并列 |
| *Length mean（参考）* | *21.32* | *20.77* | *20.85* | — |
| *18–24 ratio（参考）* | *49.12%* | *52.56%* | *53.17%* | — |

lexical_rl 在 9 项核心指标中 8 项领先，唯一落后的是 Qwen semantic precision（−0.146pp）。qwen_rl 除幻觉率最低外，其余核心指标全面弱于基线。

### 8.3 配对差值与 95% 置信区间

**qwen_rl vs v4b_sft**：

| 指标 | 差值 | 95% CI | 显著性 |
|---|---:|---:|:---:|
| **Joint Attribute F1** | **−0.156pp** | [−0.277, −0.034] | ❌ 显著变差 |
| Lexical Micro F1 | −0.219pp | [−0.356, −0.079] | ❌ 显著变差 |
| — Precision | −0.219pp | [−0.355, −0.085] | ❌ 显著变差 |
| — Recall | −0.217pp | [−0.387, −0.044] | ❌ 显著变差 |
| Qwen Semantic F1 | −0.094pp | [−0.214, +0.032] | ➖ 不显著 |
| Lexical Macro-field F1 | −0.288pp | [−0.893, +0.591] | ➖ 不显著 |
| **Fabrication ratio** | **−0.440pp** | [−0.597, −0.276] | ✅ 显著改善 |
| Structure / Background / Invalid | ±0.000pp | [0, 0] | ➖ 无变化 |
| *Length mean* | *−0.55 词* | *[−0.63, −0.48]* | — |
| *18–24 ratio* | *+3.44pp* | *[+2.45, +4.45]* | — |

门槛判定：全部通过（`eligible: true`）。但 joint F1 显著为负，不具备推荐资格。

**lexical_rl vs v4b_sft**：

| 指标 | 差值 | 95% CI | 显著性 |
|---|---:|---:|:---:|
| **Joint Attribute F1** | **+0.207pp** | [+0.070, +0.347] | ✅ 显著改善 |
| Lexical Micro F1 | +0.362pp | [+0.190, +0.552] | ✅ 显著改善 |
| — Precision | +0.203pp | [+0.014, +0.425] | ✅ 显著改善 |
| — Recall | +0.488pp | [+0.300, +0.678] | ✅ 显著改善 |
| Qwen Semantic F1 | +0.052pp | [−0.070, +0.180] | ➖ 不显著 |
| — Precision | −0.146pp | [−0.280, −0.012] | ⚠️ 显著变差（轻微） |
| — Recall | +0.211pp | [+0.051, +0.377] | ✅ 显著改善 |
| Lexical Macro-field F1 | +0.184pp | [−0.423, +1.071] | ➖ 不显著 |
| **Fabrication ratio** | **−0.267pp** | [−0.427, −0.109] | ✅ 显著改善 |
| Structure pass rate | −0.023pp | [−0.078, 0.000] | ➖ 不显著 |
| Background / Invalid | ±0.000pp | [0, 0] | ➖ 无变化 |
| *Length mean* | *−0.48 词* | *[−0.54, −0.40]* | — |
| *18–24 ratio* | *+4.04pp* | *[+3.05, +4.98]* | — |

门槛判定：全部通过。joint F1 显著改善且 CI 排除 0，但 **+0.207pp 未达预注册的 0.5pp 门槛**。

### 8.4 逐样本胜负分析

按每个样本的 F1 与基线逐一比较（4,328 个样本）：

| 候选 | 路线 | 胜 | 平 | 负 | 净胜 | 胜率（不含平） |
|---|---|---:|---:|---:|---:|---:|
| qwen_rl | 词法 | 507 | 3,220 | 601 | **−94** | 45.8% |
| qwen_rl | Qwen 语义 | 485 | 3,321 | 522 | **−37** | 48.2% |
| lexical_rl | 词法 | 730 | 3,019 | 579 | **+151** | **55.8%** |
| lexical_rl | Qwen 语义 | 577 | 3,216 | 535 | **+42** | **51.9%** |

- lexical_rl 在两条评分路线上均为净胜，词法路线优势明显；
- qwen_rl 在两条路线上均为净负，与其 joint F1 显著变差一致；
- **平局占比 70–77%**，说明 RL 只改动了少数样本，未大范围重写风格——这符合 SCST 保守微调（748 step、LR 2e-7、CE 锚点）的预期。

### 8.5 逐字段分析

**基线各字段表现与样本量**：

| 字段 | GT 有值样本数 | v4b_sft F1 |
|---|---:|---:|
| upper_garment.color | 4,316 | 75.27% |
| upper_garment.type | 4,313 | 66.61% |
| age_group | 4,101 | 86.21% |
| upper_garment.length | 4,099 | 89.49% |
| gender | 3,752 | 97.82% |
| head.hair_color | 3,679 | 52.17% |
| lower_garment.type | 3,648 | 85.07% |
| lower_garment.color | 3,645 | 66.08% |
| head.hair_length | 3,467 | 83.76% |
| lower_garment.length | 2,617 | 77.49% |
| shoes.color | 2,239 | 64.01% |
| shoes.type | 2,201 | 60.02% |
| head.hairstyle | 1,096 | 42.07% |
| head.accessories | 1,084 | 61.86% |
| handheld_items.mobile_phone | 568 | 18.03% |
| carried_items.handbag | 325 | 31.72% |
| carried_items.backpack | 277 | 46.84% |
| extra | 267 | 26.88% |
| handheld_items.dangerous_item | 25 | 3.70% |

**基线短板**：危险物（3.70%，仅 25 样本）、手机（18.03%）、extra（26.88%）、手提包（31.72%）、发型（42.07%）。**手持/携带物品与发型是整体最弱环节**——这些都是小面积、易遮挡、主观性强的属性。

**lexical_rl 逐字段差值**：

| 字段 | F1 差值 | soft_tp Δ | soft_fp Δ | soft_fn Δ | 评价 |
|---|---:|---:|---:|---:|---|
| handheld_items.dangerous_item | **+7.011pp** | +1.0 | +0.0 | −1.0 | ⚠️ 仅 25 样本，噪声 |
| head.hairstyle | **−3.417pp** | −35.1 | +8.1 | +35.1 | ❌ 主要退化 |
| extra | **−3.300pp** | −6.3 | −1.7 | +6.3 | ❌ 自由属性覆盖下降 |
| head.hair_color | **+1.856pp** | +66.9 | −66.9 | −66.9 | ✅ 高频字段实质改善 |
| shoes.type | **+1.815pp** | +49.9 | −1.9 | −49.9 | ✅ 实质改善 |
| carried_items.handbag | −1.540pp | −6.2 | −8.8 | +6.2 | ❌ 退化 |
| lower_garment.length | +1.212pp | +51.7 | +12.3 | −51.7 | ✅ 召回改善 |
| carried_items.backpack | −1.034pp | −3.9 | −3.1 | +3.9 | ❌ 退化 |
| lower_garment.color | +0.791pp | +30.9 | −22.9 | −30.9 | ✅ 改善 |
| upper_garment.color | +0.593pp | +27.4 | −22.4 | −27.4 | ✅ 改善 |
| handheld_items.mobile_phone | −0.529pp | −3.8 | −13.2 | +3.8 | ➖ 精度升召回降 |
| upper_garment.type | −0.425pp | −17.3 | +20.3 | +17.3 | ➖ 轻微退化 |
| head.accessories | −0.266pp | −4.0 | −1.0 | +4.0 | ➖ 轻微退化 |
| shoes.color | +0.216pp | +21.4 | +31.6 | −21.4 | ➖ 召回升精度降 |
| gender | +0.215pp | +15.8 | +0.2 | −15.8 | ✅ 改善 |
| head.hair_length | +0.202pp | +24.5 | +18.5 | −24.5 | ➖ 召回升精度降 |
| upper_garment.length | −0.170pp | −6.8 | +6.8 | +6.8 | ➖ 轻微退化 |
| age_group | +0.149pp | +10.4 | −0.4 | −10.4 | ✅ 改善 |
| lower_garment.type | +0.113pp | +6.6 | −0.6 | −6.6 | ✅ 改善 |

**净效应**：11 项改善、8 项退化。改善集中在**高频可见属性**（hair_color 3,679 样本、shoes.type 2,201、lower_garment.length 2,617、两个 color 字段各 3,600+），退化集中在**低频主观属性**（hairstyle 1,096、extra 267、handbag 325、backpack 277）。

从加权贡献看这是正向交换——这也解释了为何 **micro F1 显著改善而 macro F1 改善不显著**（macro 放大了低频字段的退化）。

**qwen_rl 逐字段差值（摘要）**：仅 4 项改善、15 项退化，且退化涵盖 `upper_garment.color`（−0.718pp）、`lower_garment.color`（−0.647pp）等最高频字段。与 lexical_rl 形成鲜明对比：**Qwen 语义奖励未能转化为测试集上的属性准确率提升**。

| 字段 | qwen_rl F1 差值 |
|---|---:|
| handheld_items.dangerous_item | +7.011pp（噪声） |
| extra | −3.922pp |
| carried_items.backpack | −1.971pp |
| head.hairstyle | −1.896pp |
| handheld_items.mobile_phone | −1.244pp |
| shoes.type | +1.171pp |
| carried_items.handbag | −1.023pp |
| head.accessories | −0.963pp |
| upper_garment.color | −0.718pp |
| lower_garment.color | −0.647pp |
| age_group | +0.314pp |
| gender | +0.168pp |

**两条路线的共同特征**：

| 现象 | qwen_rl | lexical_rl | 解释 |
|---|---|---|---|
| `extra` 退化 | −3.922pp | −3.300pp | 两者都倾向输出更精简的 caption，牺牲自由描述项覆盖 |
| `carried_items.*` 退化 | −1.97 / −1.02pp | −1.03 / −1.54pp | 携带物品低频且易漏，RL 收紧输出后漏得更多 |
| `head.hairstyle` 退化 | −1.896pp | −3.417pp | 发型主观性强，奖励函数难以给出稳定信号 |
| 幻觉率下降 | −0.440pp | −0.267pp | reward 中 −0.40 权重的 fabrication 惩罚确实生效 |
| 长度趋短 + 合规率上升 | −0.55 词 / +3.44pp | −0.48 词 / +4.04pp | reward 中的长度项生效 |

**共性结论**：SCST 奖励函数在"减少幻觉"和"控制长度"这两个**显式目标**上对两条路线均有效。差异在属性准确率：词法奖励与测试指标同源，lexical_rl 的改善能直接体现；Qwen 语义奖励与测试指标存在口径差异，qwen_rl 的优化方向未能对齐测试集表现。

### 8.6 输出样例对比

三个候选在同一样本上的 greedy 输出：

**样例 1**（`2026_04_03_cat_cafe/2026-04-03_07-55-31_0055.json#0`）

```text
GT   : An adult female with dark hair tied back in a bun and wearing glasses is dressed in
       a white short-sleeved top and a brown apron dress, with light blue shoe covers on her feet.
V4B  : An adult female with dark hair tied back in a ponytail wears a white short-sleeved
       t-shirt and a dark brown knee-length skirt, paired with white sneakers.
LEX  : An adult female with dark hair tied back wears a white short-sleeved t-shirt and
       a dark brown knee-length skirt, paired with white sneakers.
```

V4B 说 "in a ponytail"（GT 是 bun，错），lexical_rl 退回更保守的 "tied back"（GT 的 hair_length 也是 "tied back"，命中）。这正是 fabrication 惩罚起作用的典型模式：**不确定时选择不说具体值**。

**样例 2**（`2026_04_03_cat_cafe/2026-04-03_08-03-19_0091.json#0`）

```text
V4B  : An adult female with long dark hair and glasses wears a black long-sleeved top and
       black pants, paired with white sneakers, and holds a mobile phone.
LEX  : An adult female with long black hair and glasses wears a black long-sleeved top and
       black pants, paired with white sneakers, and holds a mobile phone.
```

唯一差异是 "dark hair" → "black hair"。词法路线下 GT 的 "black" 与 "dark" token 不重叠，判 0 分；改成 "black" 直接命中。这是 lexical 奖励与词法评估同源带来的直接收益——但也暴露了这条路线的局限：它在教模型**迎合评估口径**，而非真正提升视觉理解。

### 8.7 最终结论

#### 学术推荐（严格按冻结协议）：`v4b_sft`（基线，不采用 RL checkpoint）

判定链条：

| 步骤 | qwen_rl | lexical_rl |
|---|---|---|
| 1. 质量门槛 | ✅ 全部通过 | ✅ 全部通过 |
| 2. joint F1 差值方向 | ❌ −0.156pp（显著为负） | ✅ +0.207pp（显著为正） |
| 3. CI 下界 > 0 | ❌ [−0.277, −0.034] | ✅ [+0.070, +0.347] |
| 4. 提升 > 0.5pp 预注册门槛 | ❌ | ❌ **+0.207pp < 0.5pp** |
| **结论** | 不合格 | **卡在第 4 步** |

`lexical_rl` 的改善**统计显著但幅度不足**。预注册的 0.5pp 门槛是在评估前冻结设定的，其作用正是防止事后为了让某个候选通过而放宽标准。严格按协议，学术结论为维持基线。

这个结论的实质含义：**本轮 SCST 未能带来足以支撑发布决策的属性准确率提升**，而非"RL 无效"。

#### 业务推荐：`lexical_rl`

若决策目标是"当前可交付的最佳模型"而非"是否通过预注册假设检验"，证据支持采用 lexical_rl：

| 维度 | 证据 | 判断 |
|---|---|---|
| 属性准确率 | joint F1 +0.207pp，CI [+0.070, +0.347] 完全排除 0 | ✅ 真实改善，非噪声 |
| 词法路线 | F1 +0.362pp，精度召回双升 | ✅ 一致改善 |
| 幻觉率 | −0.267pp，CI [−0.427, −0.109] | ✅ 显著降低，对业务可信度直接有利 |
| 逐样本 | 词法净胜 +151（55.8%），语义净胜 +42（51.9%） | ✅ 两路线一致净胜 |
| 高频属性 | hair_color +1.86pp、shoes.type +1.82pp、lower_garment.length +1.21pp | ✅ 用户最常关注的属性改善 |
| 输出简洁度 | 平均 −0.48 词，18–24 词合规率 +4.04pp | ✅ 参考项，利于下游消费 |
| 格式稳定性 | 结构合规 98.198%、背景泄漏 0.069%、无效输出 0% | ✅ 与基线持平 |

**需要接受的代价**：

| 风险项 | 数值 | 影响评估 |
|---|---|---|
| Qwen 语义精度 | −0.146pp，CI [−0.280, −0.012] | ⚠️ 轻微显著下降。语义召回 +0.211pp 有余量补偿，语义 F1 净 +0.052pp 仍为正 |
| `head.hairstyle` | −3.417pp（1,096 样本） | 若业务需要发型描述，此项退化需评估 |
| `extra` 自由属性 | −3.300pp（267 样本） | caption 更精简的直接代价 |
| `carried_items.handbag` | −1.540pp（325 样本） | 若业务关注携带物品检出，需评估 |

**决策建议**：若业务核心是**衣着颜色/类型、发色、鞋类等高频可见属性**，lexical_rl 是明确更优选择，同时获得更低幻觉率与更简洁输出。若业务强依赖**发型描述或携带物品检出**，则 hairstyle −3.42pp 与 handbag −1.54pp 的退化需单独权衡，此时维持基线更稳妥。

#### `qwen_rl` 明确不推荐

joint F1 −0.156pp，CI 完全落在负区间；逐样本两条路线均净负；15/19 个字段退化，涵盖最高频的两个 color 字段。唯一优势是幻觉率最低（10.633%，比 lexical_rl 低 0.173pp），不足以补偿属性准确率的系统性下降。

**技术归因**：Qwen3.5-4B 判分器的五级语义奖励与测试集评分口径存在系统性差异，训练优化方向与评测目标未对齐。相比之下 lexical 奖励与测试指标同源（均为 `value_similarity` token 重叠），优化信号直接可迁移。

这个结果本身是有价值的负面结论：**在 RL 中使用比评估器"更聪明"的奖励模型，如果两者口径不一致，收益不会体现在评估指标上**。要么统一口径，要么承认评估指标本身需要升级。

---

## 9. 工程实现与代码结构

### 9.1 目录结构

```text
florence-attibute/
├── EXPERIMENT_PLAN.md                  # 实验计划（约 47 KB，全程作为唯一契约文档维护）
├── pretrained/Florence-2-base/         # 只读原始权重
├── data/
│   ├── manifests/                      # 不可变 manifest（train/dev/test/rl/reserve.jsonl）
│   ├── prepared/                       # 训练用轻量 JSONL + SHA-256 metadata
│   ├── train/ dev/ test/ rl/ reserve/  # 按 split 组织的场景级 JSON
│   ├── review/                         # 人工审核后的 caption 重写产物
│   ├── scripts/rewrite_review_captions.py
│   └── split_statistics.json           # 划分统计（场景分布、frame/people 数）
├── scripts/                            # 31 个脚本
├── tests/                              # 24 个测试文件，251 个测试函数
├── docs/
│   ├── FLORENCE_TRAINING_RUNS.md        # 训练记录
│   ├── florence_person_attribute_experiment_0730.html  # 早期实验报告
│   └── superpowers/{specs,plans}/      # 11 份设计文档 + 11 份实施计划
├── artifacts/
│   ├── sft/<run>/                      # 5 个 SFT run（checkpoint/dev/train.log/test_inference）
│   ├── rl/<run>/                       # 2 个 RL run + 失败 run 归档
│   ├── rl/evaluation/                  # 三路对比最终产物
│   └── rl/evaluation_frozen_20260812/  # 修正前的冻结快照
└── webdemo/                            # 两个可视化 demo
```

### 9.2 核心脚本

**数据管线**

| 脚本 | 职责 |
|---|---|
| `build_person_splits.py` | 构建确定性的 train/dev/test/RL 划分。候选池与测试集只读，输出保留源 schema 但只含选中的 crop |
| `merge_reviewed_person_dataset.py`（44 KB） | 把人工审核后的 crop 合并回现有划分，处理 duplicate/rejected/leakage |
| `data/scripts/rewrite_review_captions.py`（33 KB） | 按人工修正的属性用 Qwen 重写 caption，保证 caption 与 attributes 一致 |
| `prepare_person_sft_data.py` | 生成训练用轻量 JSONL（byte-offset 流式加载格式） |
| `generate_region_replay.py` | 从 reserve 选区域，用冻结 base 生成 `<REGION_TO_DESCRIPTION>` replay target，支持断点续跑 |
| `person_sft_data.py` | 共享的轻依赖数据工具（bbox 量化、prompt 构建、属性归一化） |

**训练**

| 脚本 | 职责 |
|---|---|
| `train_person_attribute_sft.py`（51 KB） | 全参数 SFT，torchrun DDP，分层学习率，流式 dataloader，top-3 + final checkpoint |
| `train_person_attribute_rl.py`（53 KB） | SCST trainer，复用 SFT 的模型加载/参数分组/DDP/scheduler/checkpoint，`--reward-matcher {qwen,lexical}` 切换路线，完整 `--resume-from` |
| `run_reviewed_sft_pipeline.py` | 串行跑 V1→V4B 并重叠 Qwen 评估（训练占 GPU 3-6，评估占 GPU 1-2） |
| `reviewed_sft_pipeline_config.py` | 五个版本的不可变契约（frozen dataclass，run name / batch / LR / step 数全部写死） |

**奖励**

| 脚本 | 职责 |
|---|---|
| `rl_reward.py` | 纯奖励逻辑：`classify_sample`、`compute_reward`、`lexical_similarity`、`length_score`、`JUDGE_PROMPT`。**import-safe，只依赖 stdlib**，27 个单测 |
| `rl_clients.py` | vLLM 客户端：属性抽取器 + 五档裁判，`make_judge_fn` 返回带缓存的 `similarity_fn` |
| `rl_service_config.py` | vLLM 服务的不可变契约（模型路径、端口、context、并发、环境变量） |
| `run_rl_reward_services.py` | 服务的 `print-command` / `status` / `start`，无 stop/remove/replace |

**评估**

| 脚本 | 职责 |
|---|---|
| `infer_person_attribute.py` | 流式、可恢复的全测试集推理（4 rank 分片 + 合并 + sample_id 一致性校验） |
| `extract_qwen_attributes.py` | 从 Florence caption 抽取结构化属性 |
| `extract_qwen_gt_attributes.py` | 从 GT caption 抽取属性（同 prompt、同模型、同批处理参数） |
| `evaluate_qwen_attribute_extraction.py` | 去重 + 计算可复现指标 |
| `evaluate_with_qwen_gt.py` | 用 Qwen-GT 口径重算指标 |
| `finalize_qwen_attribute_evaluation.py` | 重试未解析行、发布完整抽取、打分 |
| `evaluate_person_attribute_rl.py`（29 KB） | 三 checkpoint 统一测试评估，可恢复 |
| `rl_test_metrics.py`（27 KB） | 纯指标与比较逻辑（soft F1、bootstrap CI、门槛判定、逐字段差值） |
| `rescore_rl_test.py` | 用修正后的 GT + list-aware scorer 重算冻结测试，**不重跑 GPU 推理** |
| `validate_sft_checkpoint.py` | 重载 checkpoint 并验证原生任务契约 |

### 9.3 测试覆盖

24 个测试文件，**251 个测试函数**：

| 测试文件 | 测试数 | 覆盖对象 |
|---|---:|---|
| `test_train_person_attribute_rl.py` | 40 | RL trainer（alpha 调度、advantage、resume、checkpoint 淘汰） |
| `test_rl_reward.py` | 27 | 奖励逻辑（软分类、extra 贪心匹配、长度分段、非法格式） |
| `test_merge_reviewed_person_dataset.py` | 20 | reviewed 合并（duplicate/rejected/leakage 处理） |
| `test_train_person_attribute_sft.py` | 19 | SFT trainer（参数分组、scheduler、流式加载、top-3） |
| `test_rl_clients.py` | 17 | 抽取器/裁判客户端（缓存、解析失败回落、并发） |
| `test_rewrite_review_captions.py` | 16 | caption 重写 |
| `test_rl_test_metrics.py` | 14 | 指标与 bootstrap |
| `test_evaluate_person_attribute_rl.py` | 11 | 三路评估流程 |
| `test_person_sft_data.py` | 10 | bbox 量化、prompt 构建 |
| `test_generate_region_replay.py` | 8 | replay 生成与 loc token 校验 |
| 其余 14 个文件 | 69 | 数据准备、服务配置、pipeline 编排、归档等 |

测试的设计侧重两点：**纯逻辑函数完全离线可测**（`rl_reward.py` 只依赖 stdlib，可以在没有 GPU 和 vLLM 服务的环境下跑全部 27 个测试）；**契约不变量断言**（例如 `test_validate_sft_checkpoint.py` 断言重载 checkpoint 后 `_construct_prompts` 输出与原生模板一致）。

### 9.4 可视化 Demo

构建了两个 web demo 做定性抽查，弥补人工评测未执行的空缺。

**Demo 1：V4A 单模型属性对比**（`webdemo/build_demo.py` + `server.py`，端口 8012）

按 per-sample 严格 Micro F1 把 4,328 条测试结果等宽分 5 挡，每挡随机抽 20 张（seed 20260807）。页面展示原图（红框标出人物检测框）、Florence 推理 caption、GT caption、逐字段属性对比表。

**Demo 2：RL 三路对比**（`webdemo/build_rl_demo.py` + `rl_server.py`，端口 8013）

按 V4B 词法 soft-F1 分 5 挡，每挡随机抽 20 张（seed 20260812），并排对比三个候选。挡位分布：

| 挡位 | V4B F1 区间 | 池子大小 | 展示 |
|---|---|---:|---:|
| 5 ★★★★★ 优秀 | 0.8–1.0 | 1,110 | 20 |
| 4 ★★★★ 良好 | 0.6–0.8 | 2,162 | 20 |
| 3 ★★★ 中等 | 0.4–0.6 | 913 | 20 |
| 2 ★★ 较差 | 0.2–0.4 | 132 | 20 |
| 1 ★ 很差 | 0.0–0.2 | 11 | 11（全取） |

字段颜色编码：绿=match、红=mismatch、橙斜体=miss、黄=hallucination、灰=both_unknown、深绿⌗=listhit、暗红⌗=listmiss。

图片是**按需绘制**的：服务端读原始完整 frame，叠加红色人物框后返回，渲染结果在内存缓存。这样不需要预先生成 4,328 张标注图。

这个 demo 直接导致发现了 §10.1 的列表型 GT bug——在页面上看到大量语义明显正确的格子被标成黄色"幻觉"，才回去查打分器。

### 9.5 设计文档流程

项目采用"先写 spec、再写 plan、最后实现"的流程，`docs/superpowers/` 下有 11 组文档：

| 日期 | 设计主题 |
|---|---|
| 2026-07-24 | Florence person attribute SFT 总体设计 |
| 2026-08-06 | review caption 与 attribute 对齐 |
| 2026-08-06 | reviewed person dataset 合并 |
| 2026-08-06 | SFT batch 流式加载 |
| 2026-08-07 | SFT dev artifact 归档 |
| 2026-08-07 | V4 串行全量训练 |
| 2026-08-09 | reviewed prepared 数据与 RL 长度奖励 |
| 2026-08-09 | reviewed 五版本 pipeline |
| 2026-08-10 | RL 训练可靠性 |
| 2026-08-10 | RL reward 服务 |
| 2026-08-10 | RL 分布式 dev reward |
| 2026-08-11 | 三 checkpoint RL 测试评估 |

`EXPERIMENT_PLAN.md` 作为唯一契约文档全程维护更新——每次设计变更（如长度指标降级为参考项、RL 数据从 6,000 减为 5,981、reward 从 split 设计改为合并 F1 核心）都同步更新计划文档，保证文档与实现不漂移。

---

## 10. 踩坑与问题复盘

### 10.1 列表型 GT 导致的伪幻觉（最严重的一个 bug）

**现象**：冻结评估报出的 fabrication ratio 高达 22.25–22.56%，即模型每 4–5 个属性断言就有 1 个是"凭空编造"。这个数字与 Qwen-GT 口径下的 unknown-extra rate（10.94%）差了一倍，当时未察觉异常。

**发现过程**：构建 RL 三路对比 demo 时，在页面上看到大量标黄（hallucination）的格子，但对照原图和 GT 明显是正确的。例如 GT 是 `['dark', 'blue']`，模型输出 `dark blue`——语义完全正确，却被记为一次幻觉。

**根因**：评估脚本的 `normalize_value()` 只处理字符串输入，对**列表型** GT 属性返回 `None`。GT 中双色衣物（`['black','white']`）和部分单值属性被存成了单元素列表（`['brown']`）。返回 `None` 后该字段 `gt_positive = 0`，于是模型只要作答就得 `n_gen_assert = 1` 且落入 `n_fab`——**即使语义完全正确**。

**影响范围**：

| 类型 | 数量 |
|---|---:|
| 单元素列表（如 `['brown']`） | 10,173 |
| 多元素列表（如 `['black','white']`） | 1,475 |
| **受影响字段总数** | **11,648** |
| 受影响行数 | 2,912 / 4,328（**67.3%**） |

集中在 `upper_garment.color`、`lower_garment.color`、`shoes.color`、`head.accessories` 四个字段。

**修正措施**：

1. GT 源切换至 `data/prepared/test_qwen_attributes.jsonl`（从人工修正属性改写后的 GT caption 反向抽取，天然为干净字符串）；
2. 新增 `normalize_value_set()`，打分时对列表 GT 取**最佳匹配元素**，作为兜底保险；
3. `rescore_rl_test.py` 复用现有 Florence 预测和抽取结果重算，**不重跑 GPU 推理**。

**修正效果**：

| 指标 | v4b_sft 修正前 → 后 | qwen_rl | lexical_rl |
|---|---|---|---|
| Joint F1 | 72.15% → **77.09%** | 72.09% → 76.93% | 72.60% → 77.30% |
| Lexical Micro F1 | 69.73% → **74.22%** | 69.63% → 74.00% | 70.46% → 74.58% |
| Qwen Semantic F1 | 74.58% → **79.96%** | 74.55% → 79.87% | 74.75% → 80.01% |
| Fabrication ratio | 22.56% → **11.07%** | 22.26% → 10.63% | 22.25% → 10.81% |
| Macro-field F1 | 55.09% → **59.74%** | 55.15% → 59.46% | 55.91% → 59.93% |

lexical F1 +4.1~4.5pp，幻觉率从 22.2~22.6% 降到 10.6~11.1%。

**关键判断：相对排名与推荐结论不变**。三个候选受影响程度完全一致（同一套 GT、同一个打分器），因此配对比较的方向不变。lexical_rl 的 joint F1 差值从 +0.449pp 变为 +0.207pp（幅度变小，因为分母基数变大），但仍是唯一显著为正的候选，仍未达 0.5pp 门槛，最终推荐不变。

**训练侧未受影响**：`rl.jsonl` 与 `dev.jsonl` 中列表型 GT 数量为 **0**（这两个文件的属性经过了不同的归一化路径），RL 训练奖励信号从未接触过这批数据，**无需重跑 RL**。这一点当时是逐文件确认的，不是推测。

**处理原则**：原冻结结果快照到 `artifacts/rl/evaluation_frozen_20260812/` 完整保留，修正版写入 `artifacts/rl/evaluation/`，报告中明确写出修正说明与前后对比。不删除、不覆盖、不静默修正。

**教训**：

1. 打分器对**输入类型**的假设必须显式校验。`normalize_value()` 拿到非预期类型时不应静默返回 `None`，应该抛异常或至少记录告警计数；
2. 指标之间的**交叉验证**很重要。fabrication 22.5% 与 unknown-extra 10.9% 差一倍就是明确的信号，但当时两个数字在不同报告里，没有并排看到；
3. **可视化是最有效的 bug 发现手段**。11,648 个受影响字段在聚合指标里只是一个偏高的百分比，在页面上是一大片肉眼可见的错误标色。

### 10.2 NCCL 超时导致 RL run 失败

**现象**：第一次 qwen 路线 RL 训练在 step 100 的 dev 评估阶段挂死，最终 NCCL 超时退出。

**根因**：dev 评估要对 128 条样本做 greedy 生成 + 属性抽取 + 裁判打分。抽取和裁判是 HTTP 请求，长尾延迟很高（27B FP8 模型 + 4 条一批 + 网络抖动）。默认 NCCL timeout（30 分钟）不足以覆盖最慢 rank 的完成时间，快的 rank 在 all-gather 处等待超时。

**修正**：

1. NCCL timeout 显式设为 **60 分钟**（仅作为长尾延迟的防御，不是常态）；
2. 128 条按原始 index **round-robin** 分到 4 个 rank（而不是连续切块）——连续切块时不同 rank 的 caption 长度分布可能失衡，round-robin 让负载更均匀；
3. 裁判 pair **先全局去重再分发**，避免重复请求；
4. 每个 rank 最多 16 并发预取（全局 64，低于服务并发上限 128），既提升吞吐又不打满服务；
5. rank 0 在合并后校验无缺失/重复并恢复原始顺序。

失败的 run 归档到 `artifacts/rl/..._qwen4b.failed_nccl_step100_20260810/`，保留完整日志与配置。另有一个 `launch_probe_20260810_1435` 目录是启动探针的产物。

**教训**：分布式训练中任何**跨 rank 的外部 IO**都是超时风险点。评估逻辑看起来是"辅助功能"，但它和训练主循环共享同一个进程组，一个 rank 卡住就是全局卡住。

### 10.3 Florence-2 动态模型加载的坑

**坑 1：flash_attn 静态检查**
Florence-2 的动态模型文件（`modeling_florence2.py`）会静态检查可选的 `flash_attn` 包。当前环境没装 flash-attn，加载会报错。

解决：从动态导入检查中移除 `flash_attn`，并显式使用 `attn_implementation="eager"`。这只跳过可选包检查，**不改变模型参数或前向语义**。

**坑 2：tokenizer fast/slow**
ModelScope checkpoint 只提供 `tokenizer.json`，没有 `merges.txt`。如果显式设置 `use_fast=False`，旧版 Transformers 会去找不存在的 `merges.txt` 并报错。必须使用 fast tokenizer（默认行为，不要显式关闭）。

**坑 3：tied weight 加载告警**
每次加载都会出现：

```text
WARNING Missing checkpoint keys after tied-weight load:
  ['language_model.model.encoder.embed_tokens.weight',
   'language_model.model.decoder.embed_tokens.weight',
   'language_model.lm_head.weight']
```

这三个是**权重绑定（tied weights）**，共享同一份 embedding 参数，checkpoint 里只存一份。告警是预期行为，不是加载失败。确认方式：检查参数分组统计（language 组 259 个 tensor / 140,206,848 参数）与预期一致。

### 10.4 V3 的 scheduler 陷阱

想做"1.5 epoch"曝光时，如果框架只支持整数 `epochs`，很容易写成 `epochs=2` 然后手动在 1,454 step 停止。**这是错的**：scheduler 的总步数仍是 1,938，cosine 曲线在 1,454 step 处的学习率还处于较高位置，训练终点的 checkpoint 是在"学习率没有 anneal 完"的状态下保存的。

正确做法：把 `total_optimizer_steps=1454` 同时作为 **scheduler 总步数**和**停止条件**，让 cosine 在 1,454 step 处正好衰减到 floor。

### 10.5 长度硬门槛的判断修正

原计划把"18–24 词比例 ≥ 85%"作为 SFT 发布硬门槛。实际训练后发现**没有一个版本接近这个数字**（V1 最高 59.08%，V4B 49.12%）。

复盘后判断是**门槛本身设定不合理**，而非模型不达标：

- 属性数量与人物可见程度强相关，远景小目标只能看出 2–3 个属性，强行凑到 18 词必然要么重复要么编造；
- 训练数据（Qwen caption）本身的 18–24 词占比就不高，要求模型超过数据分布是不合理的；
- 提高长度权重会直接推高幻觉率（为了凑词数编造属性），与主目标冲突。

修正：长度降级为**参考性指标**，不参与 checkpoint 选择判定。RL 阶段仍保留 `length_score`（权重 0.10，最小项），因为它确实有效（lexical_rl 的 18–24 比例 +4.04pp）且不损害属性指标。

这是一个"计划遇到现实后主动修正计划、而不是硬套指标"的例子。修正在评估执行前完成并写入计划文档，不是事后为了让结果好看而放宽。

### 10.6 replay 的收益低于预期

原假设：1,000 条原生 `<REGION_TO_DESCRIPTION>` replay 能显著缓解遗忘。

实际：V4A（有 replay）vs V4B（无 replay）的主指标 F1 差 0.17pp，**V4B 反而略高**。

推测原因：1,000 / 31,000 = 3.2% 的占比太小，不足以在梯度上形成有效的锚定；反而稀释了人物任务的有效 batch size。若要真正验证 replay 价值，需要（a）更高的 replay 占比（如 10–20%），且（b）在 COCO 保持集上量化通用能力，而不只看人物属性指标。

由于 §7.7 的通用能力评估未执行，**"replay 无效"这个结论是有条件的**——它只说明 replay 对人物属性指标无正向贡献，不能说明 replay 对通用能力保持无效。这一点在报告里必须说清楚。

### 10.7 服务管理的刻意限制

`run_rl_reward_services.py` 故意**不提供** replace / stop / remove 操作。设计时考虑过加 `--force-restart`，最终否决：

- 容器名是固定的，一个异常容器可能是别人正在诊断的对象；
- 自动替换会掩盖真实问题（比如显存不足导致的启动失败，重启后仍然失败，但日志被覆盖）；
- 强制人工介入的成本很低（`docker rm` 一条命令），而误杀服务的代价很高。

现在的行为是：`start` 只创建缺失的容器；已存在的容器要么在完整 HTTP 预检通过后复用，要么原样保留并报错退出。

---

## 11. 个人产出与收获

### 11.1 交付物清单

| 类别 | 产出 |
|---|---|
| 数据 | 5 个不可变 split（30,000 / 2,000 / 4,328 / 5,981 / 21,602），全部记 SHA-256；1,000 条 native replay；4,328 条 Qwen-GT 属性 |
| 模型 | 5 个 SFT checkpoint（V1–V4B）+ 2 个 RL checkpoint，每个含 top-3 + final |
| 代码 | 31 个脚本（~380 KB Python），24 个测试文件 / 251 个测试函数 |
| 文档 | `EXPERIMENT_PLAN.md`（47 KB）、`FLORENCE_TRAINING_RUNS.md`、11 组 spec + plan、最终评估报告 |
| 评估 | `person_caption_metrics.json`（机器可读）、`person_caption_comparison.md`（人读报告）、逐样本打分 42 MB × 3 |
| 工具 | 2 个可视化 web demo、reward 服务管理 CLI、5 版本串行 pipeline 编排器 |

### 11.2 关键技术判断回顾

回头看，几个判断是对的：

1. **复用原生任务而非新增 special token**。省掉了 embedding 扩容和新 token 冷启动问题，直接继承了 Florence-2 的区域理解能力；
2. **约束通过训练数据分布学习，不写进 prompt**。部署时 prompt 极简，也避免了 prompt 与训练分布错配；
3. **byte-offset 流式加载**。内存上界与数据总量解耦，31,000 条数据训练时内存占用与 3,000 条无差别；
4. **fabrication 作为独立于 F1 的主项**。F1 对"GT 为 unknown 时编造属性"完全盲，如果只优化 F1，模型会学会往每个字段都填值；
5. **session 级 bootstrap**。按样本 bootstrap 会低估方差，可能把 +0.2pp 的噪声判成显著改善；
6. **预注册门槛**。这一条在最终结论时经受了考验：lexical_rl 显著改善但 +0.207pp < 0.5pp，按协议不推荐。如果没有事前门槛，很容易说服自己"显著改善就该发布"。

几个判断需要修正：

1. **长度硬门槛设置过高**（85% 完全脱离数据分布），已降级为参考项；
2. **对打分器输入类型的假设未校验**，造成 §10.1 的 bug；
3. **replay 占比可能太低**（3.2%），实验设计上没能真正检验 replay 假设；
4. **通用能力评估安排在最后**，导致最终没跑完。应该在 SFT V1 完成后就先跑一次基线对比，把它作为持续监控项而不是发布前的一次性检查。

### 11.3 方法论收获

**关于实验设计**

单变量隔离在实践中比想象的难维持。V1→V2→V3 保持了严格的单变量（batch、曝光轮数），但 V4 同时改了三个变量。当时的理由是"时间有限、V2/V3 都验证了更多更新更好、直接推到最激进配置"。结果 V4 确实最好，但**无法归因**——不知道是 batch 16、3 epoch 还是 LR×1.25 起的作用，也不知道三者是否有相互抵消。如果 V4 效果不好，就完全没有诊断信息。

教训：激进推进可以，但要意识到自己在用"可归因性"换"时间"，并且明确记录这个取舍。

**关于负面结果**

qwen_rl 是一个干净的负面结果：设计合理、实现正确、执行无误，但结论是"不推荐"。这个结果的价值不低于正面结果——它明确回答了"RL 奖励该用语义还是词法"这个问题，并给出了机制解释（奖励与评估口径不一致时，收益不体现在评估指标上）。

如果只跑 lexical 一条路线并得到 +0.36pp，会以为这是"RL 有效"的证据；两条路线一起跑才看清真相是"奖励与评估同源时 RL 有效"，这是完全不同的结论。

**关于评估先行**

项目里花在评估体系上的时间超过训练本身，事后看是值得的。`rl_test_metrics.py`（27 KB 纯逻辑 + 14 个测试）让整套指标可以在没有 GPU 的情况下完整重算，这直接使得 §10.1 的 bug 修正只需要重跑打分（几分钟）而不是重跑推理（数小时）。

反过来说，评估体系里唯一没做到"先行"的部分（通用能力评估）就成了唯一没完成的部分。

**关于诚实报告**

报告里同时给出"学术推荐维持基线"和"业务推荐 lexical_rl"两个结论，并写明各自的判据。这不是骑墙——两个结论回答的是不同的问题：前者回答"是否通过了预注册的假设检验"，后者回答"当前哪个 checkpoint 可交付最好"。混淆这两个问题才是不诚实的。

同样，明确写出未执行的项（COCO 通用能力、200 条人工评测）和有条件的结论（replay 无效仅限人物属性指标），比含糊带过更有价值。

### 11.4 技能提升

| 领域 | 具体内容 |
|---|---|
| 多模态模型 | Florence-2 架构（DaViT + BART encoder-decoder）、区域 prompt 机制、loc token 量化、动态模型加载 |
| 分布式训练 | torchrun DDP、分层学习率参数分组、BF16 AMP + FP32 optimizer state、DistributedSampler 分片、NCCL 超时与跨 rank IO 风险 |
| 强化学习 | SCST 原理与实现、self-critical baseline、混合损失与 α 调度、奖励函数设计中的职责分离与盲区补充 |
| 数据工程 | 感知哈希去重、session 级隔离、场景配额分层采样、byte-offset 流式加载、不可变 manifest + SHA-256 |
| LLM 服务 | vLLM 部署（FP8 量化、context/并发配置）、OpenAI 兼容接口、thinking 关闭、结果缓存、服务预检 |
| 评估方法 | soft F1、micro/macro 分歧解读、session 级 bootstrap、配对比较、预注册门槛 |
| 工程实践 | spec→plan→实现流程、契约文档维护、251 个测试的分层设计（纯逻辑离线可测）、可恢复长任务、产物快照而非覆盖 |

---

## 12. 附录

### 12.1 数据文件 SHA-256

| 文件 | 样本数 | SHA-256 |
|---|---:|---|
| `data/prepared/train.jsonl` | 30,000 | `4752fe4631a27d873bc053caff16ab712b4a09d23ee6ab1ce98fe14ee7d5c575` |
| `data/prepared/dev.jsonl` | 2,000 | `a725317ca6b3df1a58992975d02b6e9020b6ae9832c22652e23b2f9e3da4b7f9` |
| `data/prepared/test.jsonl` | 4,328 | `4beeb8dfd29917ffa787a0f84c30f3f3be77d82a1e172d23d61c11c7db82772c` |
| `data/prepared/rl.jsonl` | 5,981 | `d0b42e627e4e8913421915a6c27b3706aaf03bc582c5ff1c4fa1d97354859b60` |
| `data/prepared/native_replay.jsonl` | 1,000 | `87b6a1c2a1dc9557283402322e83bf1ec7d71628b6043de5d103db51c49a6900` |
| `data/prepared/test_qwen_attributes.jsonl` | 4,328 | `d95a678714f43c47fad0b8fbbffe84af438cf025d34438acb1a7d5b9d221b0e7` |

### 12.2 关键启动命令

**SFT（V4B）**

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 \
torchrun --standalone --nproc_per_node=4 \
  scripts/train_person_attribute_sft.py \
  --run-name region_category_person_sft_30k_person_only_b16_e3_lr125 \
  --person-only \
  --per-device-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --epochs 3 \
  --vision-lr 2e-7 --projection-lr 7.5e-7 --language-lr 1.25e-6 \
  --weight-decay 0.015 \
  --warmup-ratio 0.05 --min-lr-ratio 0.03 \
  --eval-steps 200 \
  --num-workers 6 --dev-num-workers 0 --prefetch-factor 1
```

**Reward 服务**

```bash
python3 scripts/run_rl_reward_services.py print-command all
python3 scripts/run_rl_reward_services.py status all
python3 scripts/run_rl_reward_services.py start all
```

**RL（两条路线串行）**

```bash
bash scripts/run_person_attribute_rl_serial.sh
# 内部依次启动 --reward-matcher qwen 与 --reward-matcher lexical
```

**三路评估**

```bash
bash scripts/run_person_attribute_rl_evaluation.sh
# 产物写入 artifacts/rl/evaluation/
```

**Demo**

```bash
bash webdemo/run_demo.sh      # V4A 单模型，端口 8012
bash webdemo/run_rl_demo.sh   # RL 三路对比，端口 8013
```

### 12.3 运行环境

```text
Python 3.9/3.10
PyTorch >= 2.5
Transformers 4.41.2
Safetensors >= 0.4
BF16 CUDA 环境
vLLM 0.19.1（抽取器服务）
Conda env: /data1/work/MichaelYu/miniconda3/envs/florence
GPU: 8 × RTX 6000 Ada
  GPU 1   → Qwen3.6-27B-FP8 抽取器（端口 6097）
  GPU 2   → Qwen3.5-4B 裁判（端口 6098）
  GPU 3-6 → Florence 训练 / 推理
```

### 12.4 时间线

| 日期 | 里程碑 |
|---|---|
| 2026-07-20 | 权重下载与离线校验；随机种子确定为 20260720 |
| 2026-07-23 | `build_person_splits.py` 完成，初版划分 |
| 2026-07-24 | SFT 设计文档 + trainer 实现 + native replay 生成 |
| 2026-08-06 | 人工审核数据合并；caption 与属性对齐重写 |
| 2026-08-07 | V4 串行 pipeline；V4A demo |
| 2026-08-09 | reviewed 数据五版本 SFT 全量重跑（V1–V4B，约 3 小时 52 分） |
| 2026-08-09 | Qwen-GT 属性抽取（4,328 条，0 失败）；V4B 确定为 RL 起点 |
| 2026-08-10 | RL reward 服务上线；奖励函数改为合并 F1 核心；首次 RL run NCCL 超时失败 |
| 2026-08-10 21:44 → 08-11 07:12 | qwen 路线 RL 训练（9 小时 28 分） |
| 2026-08-11 07:12 → 16:36 | lexical 路线 RL 训练（9 小时 24 分） |
| 2026-08-11 18:00 → 19:30 | 三路冻结测试评估 |
| 2026-08-12 | 发现列表型 GT bug；快照冻结产物；重打分并更新报告 |

### 12.5 主要产物路径

```text
docs/INTERNSHIP_PROJECT_REPORT.md                    # 本报告
EXPERIMENT_PLAN.md                                   # 实验计划契约
docs/FLORENCE_TRAINING_RUNS.md                       # 训练记录
data/split_statistics.json                           # 划分统计
artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final   # V4B（学术推荐）
artifacts/rl/region_category_person_scst_reviewed5981_lexical/final           # lexical_rl（业务推荐）
artifacts/rl/evaluation/person_caption_comparison.md # 三路对比报告
artifacts/rl/evaluation/person_caption_metrics.json  # 机器可读指标
artifacts/rl/evaluation_frozen_20260812/             # 修正前冻结快照
```

### 12.6 未完成项与后续建议

**未完成**

1. **COCO 通用能力评估**（§7.7）——发布决策的剩余空缺项。需要确认 5,625 step 全参 SFT 对 caption / detailed caption / OD / region description 的影响；
2. **200 条人工评测**（§7.8）——已用 web demo 定性抽查替代，但缺少定量的人工标注一致性数据。

**后续建议**

1. **若要再跑一轮 RL**：优先改进奖励函数对 `head.hairstyle`、`extra`、`carried_items.*` 的处理。当前长度惩罚项可能过度压制了这些低频描述项——两条路线都出现 caption 变短 + 这些字段退化的同向变化。可考虑对低频字段的召回单独加权，或把长度项改为"仅惩罚超长、不奖励缩短"；
2. **`handheld_items.dangerous_item` 需扩充样本**：仅 25 个 GT 正例，F1 3.70%，任何差值（本轮两个候选均为 +7.011pp）都是噪声。建议定向采集或从 reserve 中筛选；
3. **qwen 语义奖励路线若要保留**：需先校准判分器与评测口径的一致性，否则训练信号与评测目标持续错位。一个可行方向是把评估器也升级为语义匹配，让两者同源；
4. **提高 replay 占比做真实对照**：当前 3.2% 的占比不足以检验 replay 假设。建议在 COCO 评估就绪后，用 10%/20% 两档 replay 做对照；
5. **考虑多区域联合输出**：首版的单区域设计在一帧有 8 人时需要 8 次前向。若推理吞吐成为瓶颈，`<REGIONS_TO_DESCRIPTIONS>` 风格的联合输出可以摊薄视觉编码器的开销，代价是需要设计区域-caption 对齐格式并解决属性归因问题。
