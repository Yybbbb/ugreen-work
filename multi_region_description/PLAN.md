# Multi-Region Description：Florence-2 多bbox 单次推理描述实验

**日期：** 2026-07-07  
**实验目录：** `florence-caption/multi_region_description/`  
**基础模型：** `florence-caption/ugipc_1231_15words_epoch3_full_handoff/checkpoint`

---

## 背景与动机

当前使用 `REGION_TO_DESCRIPTION` 任务对图像中多个 bbox 进行描述时，每个 bbox 需要独立调用一次模型推理，导致：

- vision encoder（DaViT）对同一张图像重复计算 N 次
- 总推理时间 = N × 单次推理时间

本实验新增任务 `<REGIONS_TO_DESCRIPTIONS>`，允许在单次推理中输入多个 bbox，模型一次性输出所有描述，从而将 vision encoder 的计算从 N 次降至 1 次。

---

## 任务设计

### 新任务 Token

```
<REGIONS_TO_DESCRIPTIONS>
```

### 输入格式

多个 bbox 用 `<sep>` 特殊 token 分隔（`<sep>` 已存在于 tokenizer 的 special tokens 中）：

```
Describe the following regions: <loc_x1><loc_y1><loc_x2><loc_y2> <sep> <loc_x1><loc_y1><loc_x2><loc_y2> <sep> ...
```

bbox 坐标使用现有的 `<loc_XXX>` 量化 token（1000 个 bin，已在 tokenizer 中注册）。

### 输出格式（方案 A，主方案）

描述按输入 bbox 顺序排列，用 `<sep>` 分隔：

```
description for region 1 <sep> description for region 2 <sep> description for region 3
```

**对应关系：** 隐式顺序对应。`descriptions[i]` 对应 `input_bboxes[i]`。

**后处理：** 按 `<sep>` 拆分输出，校验数量是否等于输入 bbox 数量。

---

## 备选方案（方案 B）

> 若方案 A 在实验中出现较高的顺序错位率（>5%），切换至此方案。

**输出格式：** 描述在前、bbox token 在后交错排列，bbox 显式绑定，不使用 `<sep>` 分隔：

```
description1<loc_x1><loc_y1><loc_x2><loc_y2>description2<loc_x1><loc_y1><loc_x2><loc_y2>...
```

**为什么 description-first：** Florence-2 所有 region/grounding 任务的原生输出格式都是"文本在前、loc token 在后"（`REGION_TO_CATEGORY` → `car<loc>...`，`REGION_TO_DESCRIPTION` → `turquoise Volkswagen Beetle<loc>...`，`DENSE_REGION_CAPTION` / `OD` 同理）。base 模型本身就是在 `REGION_TO_DESCRIPTION` 上微调、已会输出 `description<loc>`，多区域只是加交错的自然推广，domain gap 最小。且描述作为主 payload 先生成，不受尾部 loc 复读出错影响，鲁棒性更好。

**为什么去掉 `<sep>`：** 描述文本中永远不会出现 `<loc>` token，因此序列天然是"文本段 + 4 个 loc"的严格交替文法，一段 loc run 的出现即等价于上一个描述结束——loc run 本身就是无歧义的分隔符，`<sep>` 是冗余的。去掉后序列更短、更贴近原生格式（相当于"bbox 由用户指定的 `DENSE_REGION_CAPTION`"）。loc 数量出错（吐了 3 或 5 个）只影响该框的坐标解码，不影响描述间切分，`<sep>` 对此也无能为力。

**后处理：** 用正则 `(.*?)((?:<loc_\d+>){4})` 迭代匹配，每次得到 (描述, 4 个 loc) → 一个 (description, bbox) 对；末尾校验对数是否等于输入 bbox 数。

**优点：** 即使顺序错乱，依然可通过解析 `<loc_XXX>` token 还原对应关系，出错是局部的而非整体错位。  
**缺点：** 输出序列更长；模型需复读输入 bbox（存在复读出错的可能）。  
**参考：** 与现有 `DENSE_REGION_CAPTION` 任务输出格式一致（文本在前、loc 在后、无分隔符），区别仅在于 bbox 由用户指定而非模型生成。

---

## 实现步骤

### Step 1：修改 processor（`processing_florence2.py`）✅ 已完成

文件：`ugipc_1231_15words_epoch3_full_handoff/checkpoint/processing_florence2.py`

> 实际实现说明：`multi_region_text` 分支在 `Florence2PostProcesser` 之前提前返回（该后处理器不认识新类型），按 `<sep>` 拆分并过滤空串。新任务 token `<REGIONS_TO_DESCRIPTIONS>` 与旧 `<REGION_TO_DESCRIPTION>` 不构成子串误匹配。

1. 在 `tasks_answer_post_processing_type` 字典中新增：
   ```python
   '<REGIONS_TO_DESCRIPTIONS>': 'multi_region_text',
   ```

2. 在 `task_prompts_with_input` 字典中新增 prompt 模板：
   ```python
   '<REGIONS_TO_DESCRIPTIONS>': 'Describe the following regions: {input}',
   ```

3. 在 `post_process_generation` 方法中新增 `multi_region_text` 分支：
   ```python
   elif task_answer_post_processing_type == 'multi_region_text':
       raw = text.replace('<s>', '').replace('</s>', '')
       descriptions = [s.strip() for s in raw.split('<sep>')]
       final_answer = {'descriptions': descriptions}
   ```

4. 在 `_construct_prompts` 中若需要特殊处理，可参考 `REGION_TO_DESCRIPTION` 的方式新增分支。

---

### Step 2：构造训练数据 ✅ 已完成

**数据来源：** `florence-data/ugipc-person-region-descriptions-yolo26m-no-name/`（`REGION_TO_DESCRIPTION` 的逐 crop 推理结果，含 `bbox_loc_0_999` 与 `text`，prompt 中不带 region name）。

**构造逻辑（`scripts/prepare_data.py`）：**

```python
# 每张图（每个 JSON）构成一个训练样本：
#   1. 保留 status == "ok" 的 region
#   2. 按描述文本去重：完全相同的描述只保留 crop_area 最大的那个 region
#   3. 去重后若 region 数 > 5，按 crop_area 保留最大的 5 个
#   4. 0 个有效 region 的图直接跳过
# 每个样本：
#   prompt = "<REGIONS_TO_DESCRIPTIONS><loc_...><sep><loc_...><sep>..."
#   label  = "description1<sep>description2<sep>..."
```

**为什么要按描述去重（关键决策）：**

源数据用 `REGION_TO_DESCRIPTION` 对每个 crop 单独推理得到。统计发现原始多区域样本（N≥2）中：

| 类别 | 占比 |
|---|---|
| 描述完全相同 | 14.6% |
| 部分描述重复 | 47.7% |
| 全部唯一 | 仅 37.7% |

原因：源模型在小 crop 上倾向输出泛化的整体场景描述、YOLO 对同一人产生重叠框、同场景路人外观趋同。若不处理，方案 A 的顺序对应会让模型学到"忽略 bbox 差异、复读同一句"的捷径，摧毁任务目标。因此对同一样本内相同描述只保留面积最大的 region。

**实际产出（使用 no-name 数据重新构造，去重后）：**

| 指标 | 值 |
|---|---|
| 总样本 | 3314 |
| Train/Test | 2983 / 331 |
| N=1 | 1270 |
| N=2 | 568 |
| N=3 | 383 |
| N=4 | 328 |
| N=5 | 765 |
| 多区域（N≥2）描述唯一率 | **100%** |
| bbox/描述数量错配 | 0 |
| 跳过（0 crop） | 378 |

- 保留 N=1 样本：让模型在单区域输入时也能正常退化工作
- 输出文件：`multi_region_description/data/ugipc-person-region-descriptions-yolo26m-no-name/train.jsonl`

---

### Step 3：微调训练

**参考：** `FLorence-ft/experiment/florence-json-train/train_florence2.py`

**关键配置：**
- `max_length`：建议设为 768（N个描述拼接后较长）
- 学习率、epoch：与现有 `REGION_TO_DESCRIPTION` 任务微调保持一致，视 loss 曲线调整
- 当前训练入口：`florence-caption/scripts/train_multi_region.py`，默认读取 no-name 构造数据的 `train.jsonl`

---

### Step 4：推理代码

```python
# 构造输入：将多个 bbox 转为 loc tokens 并用 <sep> 连接
def build_multi_region_prompt(bboxes, image_size):
    loc_strings = []
    for bbox in bboxes:
        loc_str = encode_bbox_to_loc_tokens(bbox, image_size)  # <loc_x><loc_y><loc_x><loc_y>
        loc_strings.append(loc_str)
    input_str = ' <sep> '.join(loc_strings)
    return f'<REGIONS_TO_DESCRIPTIONS>{input_str}'

# 推理
outputs = model.generate(...)
result = processor.post_process_generation(outputs, task='<REGIONS_TO_DESCRIPTIONS>', image_size=image_size)
descriptions = result['<REGIONS_TO_DESCRIPTIONS>']['descriptions']

# 数量校验
assert len(descriptions) == len(bboxes), f"描述数量({len(descriptions)})与bbox数量({len(bboxes)})不一致，需 fallback"
```

推理脚本存放于 `multi_region_description/scripts/infer.py`

---

### Step 5：评估

- **功能验证：** 对 held-out 样本检查 `len(descriptions) == len(bboxes)` 的比率（目标 >95%）
- **质量评估：** 抽样比较单次推理结果与逐个推理结果的描述质量（CLIP 相似度或人工评估）
- **效率测量：** 对比 N=3/5 时的推理时间，验证速度提升

---

## 文件结构

```
florence-caption/multi_region_description/
├── PLAN.md                      ← 本文件
├── data/
│   ├── train.jsonl              ← 旧 Step 2 产物
│   └── ugipc-person-region-descriptions-yolo26m-no-name/
│       ├── train.jsonl          ← no-name Step 2 训练集：2983 条
│       └── test.jsonl           ← no-name Step 2 测试集：331 条
├── scripts/
│   ├── prepare_data.py          ← Step 2：训练数据构造（已完成）
│   ├── train.py                 ← Step 3：微调训练（待写；当前训练入口为 ../../scripts/train_multi_region.py）
│   └── infer.py                 ← Step 4：推理代码（待写）
└── checkpoint/                  ← 微调后的模型权重（训练完成后）
```

---

## 风险与 fallback

| 风险 | 处置 |
|---|---|
| 模型少/多生成描述导致顺序错位 | 推理时数量校验，错位则逐个单独推理（fallback） |
| 顺序错位率 >5% | 切换到方案 B（bbox-description 交错输出） |
| 长序列导致描述截断 | 适当减小每个描述长度，或限制每次 N ≤ 3 |
| 微调后原任务性能下降 | 混合原 `REGION_TO_DESCRIPTION` 数据一起训练（multi-task） |
