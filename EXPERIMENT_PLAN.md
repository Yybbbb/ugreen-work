# Florence2 原生 REGION_TO_CATEGORY 人物属性 Caption SFT + RL 实验计划

## 1. 实验目标

本实验从 Florence-2-base 原始预训练权重出发，全参数微调一个基于人物区域的自然语言属性描述任务，用于后续跨镜人物检索。

最终 caption 需要同时满足：

1. 描述人物可见属性，不描述场景和背景；
2. 理想长度为 18-24 个英文单词；
3. 人物主体位于句首，整体保持单句且句式相对稳定；
4. 保留 Florence2 的开放属性发现能力，不限制为固定属性表；
5. 尽量保留 Florence2-base 的通用图像理解、caption、目标检测和区域描述能力。

期望输出示例：

```text
An adult female with long black hair wears a white jacket, black pants, and glasses while carrying a backpack.
```

最终输出不是 JSON、管道分隔字段或严格槽位模板。固定模板只允许用于小规模对照或短暂预热，主 SFT 和最后阶段必须以 Qwen 自然 caption 为主要监督信号。

## 2. 预训练权重与运行环境

### 2.1 ModelScope 权重

模型已从 ModelScope 官方镜像下载：

```text
模型 ID：AI-ModelScope/Florence-2-base
本地路径：/data1/work/MichaelYu/florence-attibute/pretrained/Florence-2-base
```

下载目录约 888 MB，包含：

```text
model.safetensors       463,221,266 bytes
pytorch_model.bin       464,421,827 bytes
config.json
configuration_florence2.py
modeling_florence2.py
processing_florence2.py
preprocessor_config.json
tokenizer.json
vocab.json
```

正式训练统一从 `model.safetensors` 加载，不同时加载 `pytorch_model.bin`。保留整个 `pretrained/Florence-2-base` 目录为只读原始副本，不直接在其中修改 processor 或权重。

已完成的离线校验结果：

| 检查项 | 结果 |
|---|---|
| Processor | `Florence2Processor` |
| Tokenizer | `BartTokenizerFast` |
| 模型类 | `Florence2ForConditionalGeneration` |
| 参数量 | 231,414,016 |
| 权重 dtype | FP16 |
| 原生区域 prompt 展开 | `What does the region <loc_1><loc_2><loc_3><loc_4> describe?` |

关键文件 SHA-256：

```text
config.json
c666d0fe0172d46e115e8fba6cd93cd83714575b33a73005cab8d24ce2a3aa8f

model.safetensors
03075d2d2d2bbd3e180b9ba0afae4aa8563226e2d32911656966e05b2f2ee060

processing_florence2.py
f146023a507c009f425a49ee39aa037f4f25c64e14336e3e4f3f1d7377a68e98
```

### 2.2 环境约束

建议基准环境：

```text
Python 3.9/3.10
PyTorch >= 2.5
Transformers 4.41.2
Safetensors >= 0.4
BF16 CUDA 环境
```

ModelScope checkpoint 只提供 `tokenizer.json`，加载时必须使用 fast tokenizer。不要显式设置 `use_fast=False`，否则旧版 Transformers 会寻找不存在的 `merges.txt`。

Florence2 的动态模型文件会静态检查可选的 `flash_attn`。当前实验不要求安装 flash-attn，加载模型时仅从动态导入检查中移除 `flash_attn`，并显式使用：

```python
attn_implementation="eager"
```

这只跳过可选包检查，不改变模型参数或前向语义。

## 3. 使用原生 `<REGION_TO_CATEGORY>` 任务

### 3.1 任务接口

本实验不新增任务、不新增 processor prompt，也不修改 tokenizer。人物样本统一使用 Florence2-base 已有的原生任务：

```text
<REGION_TO_CATEGORY><loc_x1><loc_y1><loc_x2><loc_y2>
```

下载 checkpoint 的 `processing_florence2.py` 已包含：

```python
self.tasks_answer_post_processing_type["<REGION_TO_CATEGORY>"] = "pure_text"
self.task_prompts_with_input["<REGION_TO_CATEGORY>"] = (
    "What is the region {input}?"
)
```

因此不需要修改 `processing_florence2.py`。`_construct_prompts()` 会删除任务宏并把 loc tokens 填入原生自然语言模板。

结论：

- 不新增任何其他任务名；
- 不调用 `tokenizer.add_special_tokens()`；
- 不调用 `model.resize_token_embeddings()`；
- 不修改 task prompt；
- 训练和部署都只使用原生 `<REGION_TO_CATEGORY>`。

### 3.2 模型实际接收的 Prompt

调用字符串：

```text
<REGION_TO_CATEGORY><loc_416><loc_239><loc_510><loc_523>
```

Processor 展开后，模型实际看到：

```text
What is the region <loc_416><loc_239><loc_510><loc_523>?
```

输入 prompt 中不添加 18-24 词、背景限制或具体属性列表。人物优先、自然句式、长度和无背景要求全部通过 Qwen SFT target、评测门槛及 RL reward 学习。

### 3.3 微调后的任务效果与代价

Florence2-base 的 `<REGION_TO_CATEGORY>` 原本偏向回答区域类别。经过本实验的自然 caption SFT 后，该任务会被领域化为：给定一个人物 bbox，输出 18-24 词左右的开放词表人物属性描述。

预期效果：

- 保留 Florence 原生 prompt 分布，不引入新的任务路由；
- 直接利用 `<REGION_TO_CATEGORY>` 已有的区域定位和视觉语义能力；
- 输出从短 category 扩展为人物属性自然 caption；
- 仍可发现 GT 未覆盖的发型、纹理、徽章、手表、图案和配饰。

明确代价：

- `<REGION_TO_CATEGORY>` 在非人物区域上的原始短类别能力可能退化；
- 该退化属于本方案主动接受的任务重定义，必须单独测量但不作为人物模型的硬性回退门槛；
- `<REGION_TO_DESCRIPTION>`、`<CAPTION>`、`<OD>` 等其他任务 prompt 保持不变，但由于共享全部权重，仍需通过低学习率、短训练和 replay 控制间接遗忘。

### 3.4 单区域输入约束

首版每个训练样本只描述一个人物 bbox。每帧最多保留 8 人是数据采样上限，不表示一次 prompt 输入 8 个 bbox。

同一帧中的 8 人分别形成 8 条样本：

```text
full image + person bbox 1 -> caption 1
full image + person bbox 2 -> caption 2
...
full image + person bbox 8 -> caption 8
```

首版不使用 `<REGIONS_TO_DESCRIPTIONS>` 风格的多区域联合输出，因为多区域输出还需要设计区域与 caption 的对齐格式，并会显著增加属性级 reward 的归因难度。

### 3.5 Checkpoint 保存与重载

不要修改原始 `pretrained/Florence-2-base`。每个 SFT/RL checkpoint 保存原始 processor 配置及微调模型：

```text
processing_florence2.py
preprocessor_config.json
tokenizer.json
config.json
model.safetensors
```

每次保存 checkpoint 后，必须在新进程中执行离线重载，验证：

```python
processor._construct_prompts([
    "<REGION_TO_CATEGORY><loc_1><loc_2><loc_3><loc_4>"
])[0] == "What is the region <loc_1><loc_2><loc_3><loc_4>?"
```

并验证：

```python
processor.tasks_answer_post_processing_type["<REGION_TO_CATEGORY>"] == "pure_text"
```

## 4. 完整前处理流程

### 4.1 输入图像与 bbox

每条样本读取完整 frame，人物区域使用 `expanded_bbox_xyxy`。不直接把 person crop 作为模型图像输入，crop 只用于去重、质量计算和 RL 视觉 grounding。

给定原图宽高 `(W, H)` 和像素 bbox：

```text
[x1, y1, x2, y2]
```

使用 Florence2 的 1000-bin floor 量化：

```python
loc_x = floor(x / (W / 1000))
loc_y = floor(y / (H / 1000))
loc = clamp(loc, 0, 999)
```

得到：

```text
<loc_x1><loc_y1><loc_x2><loc_y2>
```

不能直接复用其他模型生成的 0-999 bbox 字段，必须从原始像素坐标重新量化。

### 4.2 调用侧 prompt 构建

```python
raw_prompt = (
    "<REGION_TO_CATEGORY>"
    f"<loc_{x1}>"
    f"<loc_{y1}>"
    f"<loc_{x2}>"
    f"<loc_{y2}>"
)
```

raw prompt 只包含任务宏和四个 loc token，不额外拼接变化的自然语言指令。

### 4.3 Processor 展开

`Florence2Processor._construct_prompts()` 将 raw prompt 替换为：

```text
What is the region <loc_x1><loc_y1><loc_x2><loc_y2>?
```

随后：

1. `CLIPImageProcessor` 将完整 frame 转为 `768x768` 模型输入；
2. `BartTokenizerFast` 对展开后的自然语言 prompt 和 loc tokens 编码；
3. processor 返回 `input_ids`、`attention_mask` 和 `pixel_values`。

### 4.4 SFT target

SFT target 使用 Qwen 原始英文自然 caption，不使用 JSON 和固定模板。

Target tokenizer 设置：

```text
padding=True
truncation=True
max_length=64 tokenizer tokens
```

Padding token 在 labels 中替换为 `-100`，不参与交叉熵。

### 4.5 输出后处理

推理生成文本后使用：

```python
processor.post_process_generation(
    generated_text,
    task="<REGION_TO_CATEGORY>",
    image_size=(image.width, image.height),
)
```

由于任务注册为 `pure_text`，输出直接返回自然语言，不解析 bbox、polygon 或 grounding 标记。

### 4.6 离线 manifest 与训练时流式加载

原始 `data/train`、`data/dev` 和 `data/reserve` 下的 scene JSON 不在训练循环中反复解析。先使用 `scripts/prepare_person_sft_data.py` 流式扫描并生成以下已预处理 JSONL：

```text
data/prepared/train.jsonl          # 30,000 条人物样本
data/prepared/dev.jsonl            # 2,000 条 dev 样本
data/prepared/native_replay.jsonl  # 1,000 条原生 replay
```

每行只保存训练所需的轻量元数据：完整 frame 路径、Florence 原生 prompt、量化 bbox、caption label、attributes、scene 和 sample_id。不会提前保存 `768x768` pixel tensor、tokenized image 或所有图像内容，避免把 31,000 条样本展开成数百 GB 的缓存。

训练启动时对每个 JSONL 做一次顺序校验，并只建立 `(文件路径, 字节 offset, task)` 索引，不把所有 JSON 对象保留在内存。DataLoader worker 根据 sampler 给出的索引 seek 到对应 offset，读取一行后才打开该样本图像；collator 只对当前 batch 调用 processor，optimizer step 完成后释放 batch 图像和 tensor。

训练集由 person 和 replay 两个索引拼接而成，使用 `DistributedSampler(shuffle=True, seed=20260720)` 在每个 epoch 打乱并按 rank 分片；不再进行 Python 全量 row shuffle。dev 使用不打乱的 sampler，确保评测顺序稳定。每个 train worker 维护自己的懒加载 JSONL 文件句柄，不跨进程共享图像或文件对象。train 设置 `prefetch_factor=1`，每个 rank 最多由 6 个 worker 各预取一个 batch，避免 PyTorch 默认预取 2 批造成额外 CPU/共享内存压力。dev 固定 `num_workers=0`，评测时不再额外创建另一组 24 个 persistent worker。

这种方式的内存上界由 `per_device_batch_size`、processor 临时 tensor 和 worker 数决定，与 31,000 条数据总量无关；训练时不会一次性加载所有原始 JSON、图像或 caption。

## 5. 数据源与当前规模

### 5.1 原始标注与独立测试集

人物 Qwen 标注源：

```text
/nfs/public/common/luoweixing/A35-pic2word/qwen_a35_unified_20260720
```

独立测试集：

```text
/nfs/public/common/luoweixing/Person_test/pending_review
```

原始标注包含 21,260 个 frame JSON 和约 70,833 个 person crop。独立测试集包含 1,945 个 frame、4,328 个 person crop。测试集在任何训练、dev、reward 调参和 checkpoint 选择之前锁定。

### 5.2 去重后的固定训练候选池

```text
/data1/work/MichaelYu/data/a35_person_filtered_train_v1
```

该目录是本实验唯一允许使用的 person train/dev 候选池。原始 NFS 目录全程只读，筛选结果只保存过滤后的 JSON，crop 和完整 frame 仍引用原路径。

| 项目 | 数量 |
|---|---:|
| 原始 frame JSON | 21,260 |
| 原始 person crop | 约 70,833 |
| 精确剔除测试 frame | 1,945 |
| 剔除测试相似 crop | 747 |
| 剔除场景内重复 crop | 7,973 |
| 因每帧最多 8 人剔除 crop | 4,166 |
| 最终候选 frame | 18,854 |
| 最终候选 person crop | 53,602 |
| 有剩余训练数据的场景 | 18 |

另有 10 个训练 JSON 因源文件读取权限不足未进入候选池，4 个 crop 因 caption 或标注状态无效被过滤。筛选前后两个 NFS 输入目录的文件指纹一致，确认没有修改源数据。

候选分布仍然不均衡：`tradeshow` 有 29,262 人，`company_surveillance_mp4_adaptive` 有 8,880 人，因此不能按自然比例直接训练。

### 5.3 不可变数据版本

正式采样后生成不可变 manifest，记录源 JSON、完整 frame、crop、bbox、场景、session、frame order、caption、结构化 attributes、质量字段、dHash、CLIP embedding ID、split 和选择原因。

随机种子固定为 `20260720`。候选池或标注发生变化时必须生成新的数据版本，不能向正在训练的 run 追加样本。

## 6. Person 级去重与采样

### 6.1 样本有效性

人物样本必须满足：

- 位于固定训练候选池或锁定测试集；
- JSON 可读且 crop annotation 状态成功；
- 完整 frame 和 person crop 均存在且可解码；
- bbox 面积为正并能生成四个合法 loc token；
- caption 非空；
- 结构化 attributes 存在，允许部分字段为 `unknown`。

本实验将 `unknown`、`none`、`no` 和其他归一化为空的值解释为该属性不存在。
评估时它们不计入已知属性正例；RL reward 中若生成具体值，则计为 fabrication。

### 6.2 与测试集隔离

去重严格按以下顺序完成：

1. 按 `scene/frame JSON` 相对路径剔除测试集的 1,945 个 frame；
2. 对 person crop 计算 64-bit dHash 和 256 维 word/bigram 哈希文本向量；
3. 跨测试集比较时，dHash 汉明距离不超过 8 直接剔除；距离不超过 16 且 caption cosine 不低于 0.88 时剔除；
4. caption 相似度不能单独触发删除，避免误删穿常见颜色和服装的不同人物。

### 6.3 场景内去重与每帧上限

完成测试隔离后，只在同一 scene 内继续去重：

- dHash 汉明距离不超过 10 时直接视为重复；
- dHash 距离不超过 18 且 caption cosine 不低于 0.85 时视为重复；
- 不做跨 scene 聚类，避免合并外观相似但身份不同的人物。

每帧最多保留 8 人。超过上限时优先保留清晰、无遮挡、截断少、检测置信度高且 crop 面积较大的样本。

CLIP 不参与上述基础去重。冻结的 CLIP 只用于后续 SFT/RL 采样中的视觉新颖性；RL reward 不再使用 CLIP（改用 Qwen 抽取器 + 相似度函数，见 §11）。

### 6.4 SFT、dev 与 RL 采样

先按完整 session 划分 dev 和 SFT，任何 session、frame 或 person sample 都不能跨 split。Person test 全部来自独立测试目录，不再从训练候选池重复切 test。

采样依据为：

- 优先保留置信度高、人物清晰、crop 较大的样本；
- 平衡 scene、tiny/small/medium/large、视角、遮挡和截断；
- 眼镜、帽子、口罩、头盔、背包、手提包、手机、手持物及开放 `extra` 属性提高优先级；
- 保留自然 caption 的长度和表达多样性，不只选择 18-24 词样本；
- 不进行有放回采样，不复制稀有样本。

定量约束：

- Person SFT train 固定 30,000 人，覆盖约 14,000-16,000 个 frame；
- `tradeshow` 和 `company_surveillance_mp4_adaptive` 各取 7,500 人，各占 SFT 的 25%；
- 其他场景合计 15,000 人，按 `sqrt(eligible_scene_people)` 和实际容量分配；
- tiny crop 不超过 25%；
- Person dev 从训练候选池按完整 session 选择 2,000 人。为同时保持 30,000 SFT 和 25% 场景上限，dev 主要使用两个超大场景的富余 session，checkpoint 选择以 scene-macro 指标为主；
- Person RL train 在 reviewed 发布后保留 5,981 人，保持为 SFT train 的严格子集，不为补足旧 6,000 配额而复制或新增样本。

## 7. 最终数据规模

| 数据集 | 人物/任务样本数 | 预计 frame/图像数 | 用途 |
|---|---:|---:|---|
| Person SFT train | 30,000 | 14,000-16,000 | 主自然 caption SFT |
| Person dev | 2,000 | 约 800-1,200 | checkpoint 选择、reward 校准 |
| Person test | 4,328 | 2,432 | reviewed 后锁定自动测试 |
| 人工测试子集 | 约 200 | 150-200 | 从 Person test 分层抽取，最终人工核验 |
| Person RL train | 5,981 | 3,368 | reviewed 后 SFT train 的高多样性子集 |
| Person reserve | 约 21,602 | 6,599 | 不作为人物 caption 监督，其中 1,000 人用于 native replay |
| Native replay | 1,000 | 1,000 | 从 reserve 选取不同 frame，保持原生区域描述能力（仅 SFT 使用） |

SFT 优化器实际看到 31,000 条样本：30,000 条人物 caption 加 1,000 条原生 replay。reviewed 后 RL 的 5,981 人严格复用当前 SFT train，不引入新人物，也不补齐旧版本减少的 19 人；RL 阶段不再使用任何 native replay，CE 锚点只来自这 5,981 条人物 caption 的 teacher-forced 损失。

Person test 的 4,328 人全部保留，不参与 dev 或 reward 调参。训练候选池中除 30,000 SFT 和 2,000 dev 外，其余约 21,602 人进入 reserve；reserve 不作为人物属性 caption 监督，仅从中取 1,000 个不同 frame 的人物区域生成 native replay。

## 8. 原生任务 Replay

为降低全参数微调遗忘，在 SFT 中混入 1,000 条 Florence2-base 原生区域描述任务：

| 原生任务 | 数量 |
|---|---:|
| `<REGION_TO_DESCRIPTION>` | 1,000 |

replay 只从 `data/manifests/reserve.jsonl` 选择，随机种子固定为 `20260720`。`company_surveillance_mp4_adaptive` 和 `tradeshow` 各取 500 个不同 frame，每帧至多一个人物区域。使用完整 frame、人物 `expanded_bbox_xyxy` 和原生 `<REGION_TO_DESCRIPTION>` prompt，由冻结的 `pretrained/Florence-2-base` greedy 推理生成 target。生成过程离线完成并支持断点续跑，正式训练不同时加载第二个 base 模型。

原始 base 的区域描述输出可能是描述文本后附四个输入区域 loc token，也可能在困难小目标上退化为 `person<loc_...>`。Replay 保留该原生输出格式；loc token 必须合法并与输入 bbox 完全一致，空输出、malformed、越界或错位 loc token 不得进入正式 replay。

人物样本使用原生 `<REGION_TO_CATEGORY>` prompt；replay 只使用 `<REGION_TO_DESCRIPTION>`。Replay 不包含 `<REGION_TO_CATEGORY>`，避免原始短类别 target 与人物自然 caption target 在同一任务上产生直接冲突。

## 9. SFT 实验

### 9.1 Florence2-base 基线

在 person dev/test 上评测：

- 原生 `<REGION_TO_CATEGORY>`；
- greedy decoding；
- beam size 3；
- 输出长度、句式、背景泄漏、已知属性和开放属性。

同时在 COCO 保持集上评测原生 caption、detailed caption、OD、region description 和 region category，作为遗忘基线。其中 region category 单独报告，不计入通用能力硬门槛，因为该任务会被主动重定义。

### 9.2 主 SFT

主实验一次性混合：

```text
30,000 条 Qwen 自然人物 caption
+ 1,000 条 Florence 原生 REGION_TO_DESCRIPTION replay
```

所有参数参与训练，仅训练 1 epoch。默认不启动第二个完整 epoch。

| 参数组 | 峰值学习率 |
|---|---:|
| 视觉编码器 | `1e-7` |
| 视觉语言投影层 | `5e-7` |
| 文本 encoder-decoder | `1e-6` |

其他配置：

| 配置 | 数值 |
|---|---:|
| 精度 | BF16 AMP，FP32 optimizer state |
| GPU | 4 张 RTX 6000 Ada |
| 每卡 micro-batch | 4 |
| 梯度累积 | 4 |
| Global batch | 64 |
| Optimizer steps | 约 485 |
| 优化器 | AdamW，β=(0.9, 0.999)，eps=`1e-8` |
| Weight decay | `0.01` |
| Label smoothing | `0.05` |
| 梯度裁剪 | `1.0` |
| Warmup | 总 step 的 5%，约 25 step |
| Scheduler | cosine decay，最低为峰值 LR 的 10% |
| Target 最大长度 | 64 tokenizer tokens |
| Train DataLoader worker | 每 rank 6 个，总计 24 个 |
| Dev DataLoader worker | 每 rank 0 个，同步读取 |
| Prefetch factor | 每 worker 1 个 batch |
| 评测间隔 | 50 optimizer steps |
| checkpoint | top 3 加 final |

global batch 计算为 `4 卡 × 每卡 4 × 梯度累积 4 = 64`。31,000 条训练样本完成一个 epoch 约产生 485 次 optimizer step。batch 4 是首选配置；正式训练前必须先进行 32 条样本的四卡 smoke test。如果单卡峰值显存不足，回退为“每卡 batch 2、梯度累积 8”，保持 global batch 仍为 64，不降低有效 batch。

正式启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  scripts/train_person_attribute_sft.py \
  --run-name region_category_person_sft_30k_replay1k_b64 \
  --per-device-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --vision-lr 1e-7 \
  --projection-lr 5e-7 \
  --language-lr 1e-6 \
  --warmup-ratio 0.05 \
  --min-lr-ratio 0.1 \
  --eval-steps 50 \
  --num-workers 6 \
  --dev-num-workers 0 \
  --prefetch-factor 1
```

启动时必须确认四个 rank 均已创建、每张卡均加载一个模型副本；global batch 大小本身不会创建 GPU 进程，进程数量由 `torchrun --nproc_per_node=4` 决定。正式模式会拒绝 `WORLD_SIZE != 4` 或有效 global batch 不等于 64 的启动，避免误用单卡或错误累积步数。485 step 的主训练预计在 50、100、…、450 step 评测 9 次，自动保留其中最优的 3 个 checkpoint，并在训练结束保存 `final/`。

仅当以下条件全部满足时追加 0.25 epoch，并将所有学习率减半：

- 第一 epoch 最后 20% step 中 person dev 属性指标仍上升；
- dev 幻觉率没有增加；
- 通用综合指标下降小于 1%。

### 9.3 模板预热仅作为条件对照

已检查的 Qwen caption 中 99.94% 是单句，99.96% 使用 wear/wearing 结构。因此默认不做模板阶段。

只有主 SFT 出现以下任一情况才运行模板对照：

- 单句率低于 98%；
- 人物主体开头率低于 95%。

对照 run 使用 0.25 epoch 自动模板预热，再用 0.75 epoch 原始 Qwen caption。最后且更重的阶段仍是自然 caption SFT。

### 9.4 本次 run 诊断与下一版超参数

0730 报告提供了可比的全参参考：Florence2-base 从头训练的 expanded 分支使用 batch 8、`1e-6`、5 epoch（12,000 条数据共 7,500 次更新），最终 Micro F1=`0.7456`、Macro-field F1=`0.6402`，是四个分支中总体属性恢复最好的设置。报告中 clean/full 分支的 loss 在最后一个 epoch 仍下降，没有证据证明 `1e-6` 本身过大；但报告没有中间 epoch 的 dev 指标，不能直接据此增加 epoch。

本次 `region_category_person_sft_30k_replay1k_b64` 使用 31,000 条样本、global batch 64，完整 epoch 只有 485 次 optimizer update。训练日志显示 dev loss 从 step 50 的 `2.3693` 降到 step 450 的 `1.6248`，没有反弹；因此首要问题是更新次数偏少。另一方面，caption 结构在 step 100 已较好（平均 20.55 词、18-24 词占比 71.9%），到 step 450 平均长度升至 21.79 词而 18-24 词占比降至 58.6%，说明继续优化时应同时监控长度和幻觉，不能只追求 token loss。

下一版建议先做可归因的 global-batch 主控：

| 配置 | 本次 | V2 主控建议 |
|---|---:|---:|
| 每卡 micro-batch | 4 | 4 |
| 梯度累积 | 4 | 2 |
| Global batch | 64 | 32 |
| 每 epoch optimizer steps | 485 | 约 969 |
| 视觉编码器 LR | `1e-7` | `1e-7`，保持保守 |
| 投影层 LR | `5e-7` | `5e-7`，先不改变 |
| 文本 encoder-decoder LR | `1e-6` | `1e-6`，先不改变 |
| epoch | 1 | 1 |

这样保持每个样本只看一次、避免过拟合，同时把更新次数约翻倍，先隔离 batch 影响。如果 V2 主控的 dev 属性指标在前 0.5 epoch 仍低于 token-loss 最优点，再运行一个唯一变量为学习率的 V2-LR+ 对照：文本 LR=`1.25e-6`、投影 LR=`6.25e-7`，视觉 LR 仍为 `1e-7`。不建议直接把视觉 LR 提高到 `1e-6`；0730 的 uniform LR 结果不能证明视觉参数在当前 31k 场景上需要同样步长。

V2 评测按样本数而不是 optimizer step 对齐：global batch 32 时每 100 step 评估一次（约 3,200 个训练样本），保留 dev 联合指标最优的 3 个 checkpoint 与 final。联合指标至少同时满足属性 Micro F1、Macro-field F1、18-24 词比例、背景泄漏率和 ungrounded 属性率；dev loss 只作辅助，不单独决定 checkpoint。若 V2 在 step 300-500 已出现属性/格式平台期，则不追加第二 epoch。

如果 V2 的属性 recall 或 Macro-field F1 在接近 1 epoch 时仍持续上升，但长度/幻觉尚未恶化，运行 V3 中间曝光实验：

| 配置 | V3 建议 |
|---|---:|
| 每卡 micro-batch | 4 |
| 梯度累积 | 2 |
| Global batch | 32 |
| 有效 epoch | 1.5 |
| Optimizer steps | 约 1,454 |
| 视觉编码器 LR | `1e-7` |
| 投影层 LR | `5e-7` |
| 文本 encoder-decoder LR | `1e-6` |
| Scheduler | 覆盖 1,454 step 的 cosine，最低为峰值 LR 的 10% |
| 评测间隔 | 100 optimizer steps |

V3 不直接跑完整 2 epoch；正式实现应以 `total_optimizer_steps=1454` 为 scheduler 和停止条件，使每条样本平均曝光 1.5 次。若暂时只支持整数 `epochs`，必须增加等价的 step 上限并同步 scheduler 总步数，不能用 `epochs=2` 配合未调整的 1,938-step scheduler。V3 仍使用联合 dev 指标早停；一旦长度 18-24 词比例下降超过 10 个百分点、背景/ungrounded 属性率上升或 Macro-field F1 连续两次评估不再提升，保留此前最佳 checkpoint，不继续追加曝光。

## 10. RL 数据与训练

reviewed 发布后的 RL manifest 从当前 SFT train 保留 5,981 人，保持相同场景配额，额外优先属性较丰富、稀有配饰、开放 `extra` 属性及不同尺度。它是 train 的严格子集，不补齐旧版本减少的 19 人，也不能只选与结构化 GT 完全一致的简单样本。

采用全参数 Self-Critical Sequence Training：

| 配置 | 数值 |
|---|---:|
| RL 数据 pass | 1 |
| Global batch | 8 |
| Optimizer steps | 748 |
| 文本/投影学习率 | `2e-7` |
| 视觉编码器学习率 | `5e-8` |
| Sampling temperature | `0.8` |
| Top-p | `0.95` |
| Baseline | 当前 policy 的 greedy 输出 |
| 评测间隔 | 100 optimizer steps |
| RL 初始化 | `artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final`（V4B） |
| 输出目录 | `artifacts/rl` |

混合损失：

```text
L = (1 - alpha) * L_CE + alpha * L_SCST
```

`alpha` 在 748 个 optimizer step 上从首步 0.10 线性增加到末步 0.50。`L_CE` 是 5,981 条人物 caption 的 teacher-forced 交叉熵（不再使用 native replay）；CE 保持主锚点，防止 reward 把模型压缩成属性列表，同时后半程提高 SCST 的影响。

每 100 step 对全部 2,000 条 dev 计算分布式 teacher-forced CE，并对固定的前 128 条 dev 运行 greedy caption reward。128 条记录按原始 index round-robin 分到四个 rank，完成后由 rank 0 校验无缺失/重复并恢复原始顺序。属性抽取按每请求 4 条分块；Qwen 裁判 pair 先去重，再由每个 rank 最多 16 并发预取（全局最多 64，低于服务并发 128）。NCCL timeout 固定为 60 分钟，仅作为长尾延迟的防御措施。RL checkpoint 按平均 greedy dev reward 降序、dev CE 升序、step 升序选择 top-3；训练循环正常退出时始终额外保留 `final/`，不受 top-3 淘汰影响。

所有 checkpoint 保存 model、processor、optimizer、scheduler、GradScaler、epoch、下一 micro-batch、global step、top-3 元数据及每个 rank 的 Python/CPU/CUDA RNG。`--resume-from` 从原 run 目录恢复这些状态并跳过已经消费的 batch；配置、数据 hash、alpha、生成参数或 reward 服务/模型不一致时拒绝恢复。

reward 管线依赖一个冻结的 Qwen 抽取器 + 一个相似度函数，不再使用 CLIP 或文本 embedding 模型（详见 §11）：

- 属性抽取器 `Qwen3.6-27B-FP8`（`models/Qwen3.6-27B-FP8`，caption → 固定属性表 JSON）；
- 相似度函数二选一（`--reward-matcher`）：
  - `qwen`：语义裁判 `Qwen3.5-4B`（`models/Qwen3.5-4B`），输出 5 档相似度 `{1.0, 0.75, 0.5, 0.25, 0.0}`；
  - `lexical`：词法 `value_similarity`（token 重叠，与评估器 soft F1 同款），无需裁判服务。

两者均经 vLLM 服务、`thinking disabled`、`temperature=0`，版本与 prompt 在 RL 开始前固定 SHA-256；policy 初始权重固定取 V4B `artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final`。

正式 reward 服务固定如下：

| 角色 | 模型 / 地址 | GPU | vLLM 容量与约束 |
|---|---|---:|---|
| 属性抽取 | `Qwen3.6-27B-FP8` / `http://127.0.0.1:6097/v1` | 1 | vLLM `0.19.1`，context 65,536，并发 64，单批 token 8,192；`VLLM_TEST_FORCE_FP8_MARLIN=1`，避免 Ada 上 block-FP8 Triton 非法访存 |
| 值裁判 | `Qwen3.5-4B` / `http://127.0.0.1:6098/v1` | 2 | context 32,768，并发 128，单批 token 65,536，swap 8 GiB |

两个服务均使用纯文本模式、`enable_thinking=false`。启动与检查入口为：

```bash
python3 scripts/run_rl_reward_services.py print-command all
python3 scripts/run_rl_reward_services.py status all
python3 scripts/run_rl_reward_services.py start all
```

`start` 只创建缺失的固定名称容器；若同名容器已经存在但异常，命令拒绝 stop/remove/replace。RL 在加载 Florence 权重前由 rank 0 检查 `/v1/models`，再发送确定性 `OK` 请求并要求 `content="OK"` 且 reasoning 为空；结果广播到全部 rank，失败时一致退出。`lexical` 路线只检查抽取器，`qwen` 路线检查两者。仅诊断场景可显式使用 `--skip-reward-service-preflight`，该状态会写入 run invariants，正式训练不得跳过。

## 11. RL 奖励函数

RL reward 是**合并的 soft-F1 核心设计**：一个相似度加权的属性 F1 作为主项（统一覆盖命中/冲突/少说/GT-extra 未覆盖），一个 fabrication 惩罚覆盖 F1 看不到的"多说"盲区，加上句式/长度/背景三个小项。**相似度函数可切换**，因此同一套 `classify_sample + compute_reward` 支持两条路线（§11.5）。不使用 CLIP、不使用文本 embedding 模型。

依赖模型：

- **属性抽取器** `Qwen3.6-27B-FP8`（`models/Qwen3.6-27B-FP8`）：`temperature=0`、`do_sample=False`、`thinking disabled`，只输入 caption 不输入图像，按固定属性表输出 JSON，未出现的字段填 `unknown`（不得省略字段）。checkpoint、tokenizer、抽取 prompt、JSON schema 在 RL 开始前固定版本并记 SHA-256；模型冻结，不参与反向传播。解析结果按 `caption_sha256 + extractor_model_sha256 + prompt_version` 缓存。
- **值匹配裁判** `Qwen3.5-4B`（`models/Qwen3.5-4B`）：仅在 `--reward-matcher qwen` 时启用，输出 5 档语义相似度。`temperature=0`、`thinking disabled`，结果按 `(field, gt_value, gen_value)` 缓存。`--reward-matcher lexical` 时不调用裁判，改用词法 `value_similarity`。

固定属性表：

```text
age_group, gender,
upper_garment.{type,color,length},
lower_garment.{type,color,length},
shoes.{type,color},
head.{accessories,hairstyle,hair_color,hair_length},
carried_items.{handbag,backpack},
handheld_items.{dangerous_item,mobile_phone},
extra[]
```

GT 经人工审核与补充，已近乎完整；SFT 阶段"多说不存在属性"的幻觉较严重，因此 reward 以打击虚构属性为主，同时用 F1 拉动属性准确率。

### 11.1 逐字段软分类（similarity-weighted）

对固定属性表的每个字段，按 GT 值与生成值分类，双方都为具体值时由 `similarity_fn(field, gt, gen)` 给出相似度 `s ∈ [0,1]`（lexical 路线为 `value_similarity`，qwen 路线为 5 档裁判分），并按评估器 soft F1 的同一公式连续化：

| GT 值 | 生成值 | 计入 |
|---|---|---|
| unknown | unknown | 忽略 |
| **unknown** | **具体** | `n_fab += 1`（多说，F1 盲区），`n_gen_assert += 1` |
| 具体 | unknown | `soft_fn += 1.0`（少说 / miss） |
| 具体 | 具体 | `soft_tp += s`，`soft_fp += 1-s`，`soft_fn += 1-s`，`n_gen_assert += 1` |

开放 `extra[]`：生成的每个 extra 项与 GT `extra[]` 用 `similarity_fn` 贪心匹配（阈值 `0.5`）：

- 命中 GT extra 项（最佳相似度 ≥ 0.5）→ `soft_tp += s`，`soft_fp += 1-s`，`soft_fn += 1-s`；
- 未命中任何 GT extra 项 → `n_fab += 1`（多说 / 虚构 extra），`n_gen_assert += 1`；
- GT 有但生成未覆盖的 extra 项 → `soft_fn += 1.0`（recall miss）。

记：

- `soft_tp / soft_fp / soft_fn`：相似度加权的 soft-F1 计数；
- `n_gen_assert`：生成的具体属性断言总数（固定字段具体值 + extra 项）；
- `n_fab`：多说数 = `GT=unknown 但生成具体` 的固定字段 + 未命中 GT extra 的 extra 项；
- `F1 = 2·soft_tp / (2·soft_tp + soft_fp + soft_fn)`（分母为 0 时 F1=0）；
- `fabrication_ratio = n_fab / max(n_gen_assert, 1)`。

**职责分离（无双重计数）**：F1 管"已知属性说对/说错/漏说/覆盖 GT-extra"；fabrication 管"凭空多说"（GT 没有却生成）。一个未匹配 extra 只进 fabrication，不进 F1 的 fp。

### 11.2 总奖励

```text
R = +0.40 * F1                              # 属性准确率（主项）
  -0.40 * fabrication_ratio                 # 多说（主项，F1 盲区）
  +0.10 * sentence_structure                # 句式
  +0.10 * length_score                      # 长度（带符号）
  -0.10 * background_penalty                # 背景
```

权重分层：F1 与 fabrication 两个主项等权（各 0.40）——前者奖励说对、惩罚说错/漏说，后者惩罚凭空多说；句式、长度、背景为最小项（各 0.10）。空输出或非法格式（空串 / JSON / 管道字段）直接 `R = -1.0`；最终裁剪到 `[-1, 1]`。

### 11.3 句式、长度与背景

句式（`sentence_structure`，0/1，纯规则）：单句、人物主体靠近句首、属性以 wear/with/carry/hold 等自然结构连接；JSON、管道字段、碎片列表得 0。

长度（`length_score`，带符号）：18-24 词 `+1.0`（贡献 `+0.10`）；12-17 或 25-28 词 `-0.5`（贡献 `-0.05`）；少于 12 或超过 28 词 `-1.0`（贡献 `-0.10`）。

背景（`background_penalty`，0/1）：caption 出现场景、建筑、位置等非人物背景关键词时为 1.0，否则 0。

### 11.4 值匹配裁判 prompt（5 档）

裁判输出 5 档相似度 `{1.0, 0.75, 0.5, 0.25, 0.0}`，直接作为 §11.1 的 `s`：

```text
You are a strict attribute-value similarity scorer for person appearance descriptions.

You receive ONE attribute field, a ground-truth (GT) value, and a generated value
(extracted from a caption). Score their similarity on a 5-point scale.

Scores:
- 1.00  exact match or synonym / different specificity of the same value
        "white" vs "light-colored" | "dark" vs "black" | "navy" vs "dark blue"
        "bright red" vs "red" | "middle-aged" vs "adult" | "shorts" vs "short pants"
- 0.75  strongly related (same category, very close)
        "jacket" vs "coat" | "pants" vs "jeans"
- 0.50  partial: a generic term weakly covering a specific one (fallback)
        "top" vs "jacket" | "top" vs "t-shirt" | "shoes" vs "sneakers"
- 0.25  weakly related (same broad category, clearly different)
        "shirt" vs "jacket" | "blue" vs "purple"
- 0.00  conflict: genuinely different value, even if same kind of attribute
        "blue" vs "red" | "male" vs "female"
        "short-sleeved" vs "long-sleeved" | "short hair" vs "long hair"

Never score above 0.00 for two values that name different things just because
they share a category. Same category but different value = 0.00.

Field: {field}
GT value: {gt_value}
Generated value: {gen_value}

Respond with ONLY this JSON object, no other text:
{"score": 1.0 | 0.75 | 0.5 | 0.25 | 0.0, "reason": "<one short clause>"}
```

解析时把 `score` snap 到最近的允许档；任何解析失败保守回落到 `0.0`（视为冲突，倾向惩罚而非放过幻觉）。

### 11.5 两条路线与对齐含义

两条路线共用 §11.1–§11.3 的全部公式，只换 `similarity_fn`：

| 路线 | similarity_fn | 特点 |
|---|---|---|
| `lexical`（`--reward-matcher lexical`） | `value_similarity`（token 重叠，与评估器 soft F1 同款） | 与评估器 F1 紧对齐；同义（white/light-colored）判 0，不奖励 |
| `qwen`（`--reward-matcher qwen`，默认） | Qwen3.5-4B 5 档语义裁判 | 同义/近义给分（white/light-colored ≈ 1.0）；与词法评估 F1 故意分歧 |

**对齐含义**：评估器报的是**词法** F1（exact / token 重叠）。lexical 路线的 F1 项与评估器同款，直接优化评估 F1；qwen 路线奖励的同义表达（white≈light-colored）评估器打 0 分，故这部分收益**不会体现在词法 F1 指标上**——这是"容忍同义"的代价。两种路线在精确匹配、冲突、少说上完全一致，只在同义/近义上分歧。最终以 F1 指标为参考，同时人工体验确认同义不被误杀。

零成本验证（128 条 SFT dev 预测，35B 作抽取+裁判替身，Spearman 排序相关性）：

| reward | vs hard_F1 | vs soft_F1 |
|---|---:|---:|
| 词法 F1 核心 | +0.68 | +0.45（hard 核心时） |
| Qwen 语义（旧 split） | +0.59 | +0.59 |
| 旧 split（词法 rule_judge） | +0.54 | +0.75 |

结论：合并 F1 核心对齐优于旧 split；词法路线对词法评估 F1 最紧，qwen 路线略低（同义分歧所致）。注：该验证用 35B 替身，真实 4B 裁判对齐可能更好。

### 11.6 实现

- `scripts/rl_reward.py`：纯奖励逻辑（`classify_sample`、`compute_reward`、`lexical_similarity`、`length_score`、`JUDGE_PROMPT`），import-safe，71 个单测覆盖。
- `scripts/rl_clients.py`：抽取器（复用 `extract_qwen_attributes` 的固定属性表 prompt）+ 5 档裁判（`make_judge_fn` 返回 `similarity_fn→float`，带缓存）。
- `scripts/train_person_attribute_rl.py`：SCST trainer（复用 SFT 的模型加载/参数分组/DDP/scheduler/checkpoint），`--reward-matcher {qwen,lexical}` 切换路线；支持完整 `--resume-from`、全量 dev CE、128 条 greedy reward、top-3 和独立 final checkpoint。
- reward 管线 + SCST 1-step 已端到端 smoke 通过（1 GPU，35B 替身）。

## 12. 评测与验收

### 12.1 人物 Caption 指标

对 Florence2-base、SFT top-3/final 以及 RL 候选使用完全相同的 checkpoint、processor、prompt 和 greedy 解码协议。Person test 只在 checkpoint 选择和 reward 阈值冻结后运行一次；dev 用于选择，test 不用于调参。总体以及按 scene、tiny/small/medium/large、视角、模糊、遮挡、IID 和 held-out session 分别报告，并以 session 为 bootstrap 单位给出 95% 置信区间：

- Qwen36-35b 或锁定的轻量 Qwen 属性抽取器使用 temperature=0、thinking disabled，只把 caption 还原为与 `attributes` 相同的 canonical JSON；抽取失败按空属性保守计 FN，不人工改写；
- 结构化属性先做规范化精确/同义词匹配，开放 `extra` 使用冻结的轻量 embedding 做 cosine 软匹配；GT 为 `unknown` 的字段不计入正负样本；
- 评估同时保存逐样本 prediction、抽取 JSON、匹配明细和失败原因，确保 Micro/Macro 汇总可复算。

主属性指标：

- 已知字段 macro precision、recall、F1；
- 全部已知属性的 Micro precision、recall、F1；
- Mean field exact（字段级完全一致率）和 19 个字段的 positive F1；
- 年龄、性别、上下衣颜色/类型/长度、鞋、发型/发色/发长、眼镜/帽子、背包/手提包、手机及开放 `extra` 的分类混淆矩阵；
- 开集 exact + semantic 属性分数；
- 平均已知匹配属性数；
- 平均新增 grounded 属性数；
- 平均冲突、Qwen extra-hallucination 和 ungrounded 属性数；
- 已知属性 recall 与支持属性 precision 的二者权衡，避免模型通过少说属性规避幻觉惩罚；
- 18-24 词比例、平均长度、P95 长度；
- 单句率、人物主体开头率；
- 背景泄漏率；
- 空输出、非法格式、特殊 token 残留率；
- 每 1,000 条 caption 的不同有效属性短语数。

建议的主 checkpoint 选择顺序为：先剔除背景泄漏率超过 1%、单句率低于 98% 或主体开头率低于 95% 的 checkpoint，再在剩余 checkpoint 中按 `0.45 * Micro-F1 + 0.25 * Macro-field-F1 + 0.15 * Mean-field-exact + 0.10 * length/structure - 0.05 * hallucination` 的 dev 联合分数选择。该分数只用于 dev，不在 test 上搜索权重。

### 12.2 小规模人工测试

从锁定 person test 中分层抽取约 200 条，覆盖不同场景、尺度和稀有属性。人工只用于最终测试，不进入训练、reward 调参或 checkpoint 选择。

人工标记：

- 正确属性；
- 可见但遗漏的属性；
- 无视觉依据的属性；
- 背景内容；
- Florence2-base、SFT 和 RL 的盲选偏好。

人工结果与自动 Qwen 指标分开报告，不用少量人工样本重新拟合阈值。人工样本应覆盖所有 scene、尺度和稀有配饰；报告每项标注的正例数，危险物等极少数属性不做过度结论。

### 12.3 通用能力保持

在互斥的 COCO 保持集上比较训练模型与 Florence2-base：

- caption CIDEr 和 CLIPScore；
- detailed caption BERTScore 和 CLIPScore；
- object detection mAP；
- region description 相对 base 输出的 BERTScore；
- non-person region category 的类别准确率和相对 base 输出一致率，仅单独报告；
- 输出长度和非法输出率。

通用综合指标只由 caption、detailed caption、OD 和 region description 组成，不包含被主动重定义的 `<REGION_TO_CATEGORY>`。每项同时报告绝对值和相对 Florence2-base 的变化；caption/detailed caption 使用 CIDEr、BERTScore、CLIPScore，OD 使用 mAP，region description 使用相对 base 的 BERTScore/CLIPScore。non-person region category 只作为单独的遗忘观察项，不进入发布总分。

### 12.4 评估执行顺序与产物

1. 锁定 checkpoint、数据 JSONL SHA-256、processor、Qwen extractor 版本和解码参数；
2. 对 dev 运行 greedy 全量评估，生成逐样本 caption、Qwen canonical tags、属性匹配和格式指标；
3. 根据 dev 联合分数保留 top-3，并对 top-3 与 final 在冻结 test 上运行一次全量评估；
4. 对通用 COCO 保持集运行 base/SFT/RL 的同一任务协议；
5. 生成 `metrics.json`、逐字段混淆矩阵、scene/scale/session 分层表、95% session-bootstrap CI 和 bad-case 列表。

Person test 当前为 4,328 个 crop、1,945 个 frame、20 个 scene；推理结果必须逐行保留 `sample_id`、原始 prompt/bbox、label、attributes、raw generation、清理后的 prediction 和 checkpoint 路径，且合并后 sample_id 必须与 test JSONL 一一对应。

### 12.5 发布门槛

SFT 必须满足：

- 18-24 词比例至少 85%；
- P95 不超过 28 词；
- 单句率至少 98%；
- 人物主体开头率至少 95%；
- 背景泄漏率不超过 1%；
- 人物属性与 grounded 属性数优于 Florence2-base；
- 通用综合指标相对 Florence2-base 下降不超过 3%。

RL 必须满足：

- 任一核心已知属性指标相对 SFT 下降不超过 1 个绝对点；
- 通用综合指标相对 SFT 下降不超过 1%；
- grounded 属性数增加，但人工判定的无依据属性不能显著增加；
- 保持全部 SFT 句式、长度和背景门槛。

若 RL 增加属性数但同时增加幻觉，先把 `grounded_attribute_count` 从 0.10 降至 0.05，并把 `qwen_extra_hallucination_penalty` 从 0.30 提高到 0.40，使用相同 5,981 条数据重跑一次。若模型明显变得过度保守、已知属性 recall 下降超过 1 个绝对点，则不继续加大惩罚，而是回退最佳 SFT。任一重试仍不通过发布门槛时，发布最佳 SFT 而不是 RL checkpoint。

### 12.6 V4B 与两条 RL 路线的冻结 test 对比

最终候选固定为 V4B SFT final、Qwen4B 语义裁判 RL final 和 lexical RL
final。三者在同一 4,328 条 test 上使用 greedy 解码；预测 caption 统一由
GPU 1 的 `Qwen3.6-27B-FP8`（6097、thinking disabled、temperature=0）抽取为
canonical 属性，再由 GPU 2 的 `Qwen3.5-4B`（6098、thinking disabled）计算
五档语义相似度。人工 reviewed `attributes` 是三者共同 GT，不复用历史旧模型
生成的 V4B 属性抽取结果。

最终报告只突出以下 8 组指标：

1. lexical Micro Precision / Recall / F1；
2. Qwen4B semantic Micro Precision / Recall / F1；
3. lexical Macro-field F1；
4. fabrication ratio（`n_fab / max(n_gen_assert, 1)`）；
5. 单句且人物主体开头的 structure pass rate；
6. 平均词数与 18-24 词比例；
7. background leakage rate；
8. empty、JSON、pipe 字段或特殊 token 构成的 invalid output rate。

每项同时报告绝对值、相对 V4B 的配对差值和以 session 为重采样单位的 2,000
次 bootstrap 95% CI。候选先通过属性 F1 不低于 V4B 超过 1 个百分点、
fabrication 不显著升高以及结构/长度/背景/格式不退化的门槛，再按
`0.5 * lexical_micro_f1 + 0.5 * qwen_semantic_micro_f1` 排序。联合 F1 提升不
超过 0.5 个百分点或其配对 CI 包含 0 时视为没有明确胜者并保留 V4B。

正式入口为 `scripts/run_person_attribute_rl_evaluation.sh`，可恢复产物统一写入
`artifacts/rl/evaluation/`。最终报告为 `person_caption_comparison.md`，机器可读
汇总为 `person_caption_metrics.json`；报告附带逐样本 lexical/semantic 胜平负和
主要字段增减，用于简要解释 checkpoint 推荐原因。

## 13. 实验矩阵

| 阶段 | Run name | 数据 | 决策目标 |
|---|---|---|---|
| Base | `florence2_base_person_eval` | dev/test | 建立人物和通用基线 |
| SFT | `region_category_person_sft_30k_replay1k_b64` | 30k + 1k | 主 checkpoint |
| SFT V2 主控 | `region_category_person_sft_30k_replay1k_b32` | 30k + 1k | global batch 32，隔离更新次数影响 |
| SFT V2 LR+ | `region_category_person_sft_30k_replay1k_b32_lr125` | 30k + 1k | 仅当 V2 主控 dev 指标支持时提高文本/投影 LR |
| SFT V3 中间曝光 | `region_category_person_sft_30k_replay1k_b32_e1p5` | 30k + 1k | global batch 32、有效 1.5 epoch，控制过拟合 |
| 条件 SFT | `region_category_person_template_warmup` | 仅句式不达标时 | 模板预热对照 |
| RL | `region_category_person_scst_reviewed5981` | 5,981，不使用 replay | 最终候选 |
| RL retry | `region_category_person_scst_reviewed5981_extra040` | 仅幻觉门槛失败时 | 降低数量奖励并提高 Qwen 多说惩罚 |

每个 run 保存：

- 完整配置；
- 代码 revision；
- 数据 snapshot hash；
- 随机种子，默认 `20260720`；
- step 级训练和评测指标；
- dev 逐样本生成结果；
- model、processor 和 tokenizer；
- 机器可读评测 JSON；
- bad-case 报告。

RL run 统一写入 `artifacts/rl/<run-name>/`。其中 `checkpoint-*` 最多保留 dev reward 最优的 3 个，`final/` 始终独立保留。

## 14. 最终产物

```text
/data1/work/MichaelYu/florence-attibute/
  pretrained/Florence-2-base/
  data/manifests/person_caption_v1/
  artifacts/baselines/florence2_base_person_eval/
  artifacts/sft/region_category_person_sft_30k_replay1k_b64/
  artifacts/sft/region_category_person_sft_30k_replay1k_b64/test_inference_final_v2/
  artifacts/rl/region_category_person_scst_reviewed5981/
  artifacts/rl/evaluation/person_caption_comparison.md
  artifacts/rl/evaluation/person_caption_metrics.json
  artifacts/reports/human_review_200.jsonl
```

最终报告必须写明实际使用的 frame 数和 person 数，而不仅是计划估算值；同时报告每一步过滤、去重和场景限额损失，并明确最终发布的是 SFT 还是 RL checkpoint。
