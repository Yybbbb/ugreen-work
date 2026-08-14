# Multi-Region Description Badcase 分析与改进建议

## 1. 背景

本分析针对以下模型与测试结果：

- 基础 checkpoint：`ugipc_1231_15words_epoch3_full_handoff/checkpoint`
- 训练数据：`qwen-person-2to5-description-loc`
- 输出格式：`description<loc_x1><loc_y1><loc_x2><loc_y2>`
- 训练集：7,492 条
- 测试集：832 条
- 人物数量范围：2～5
- 训练配置：2 epochs、batch size 16、BF16、learning rate `8e-6`、warmup 50 steps
- 最终 checkpoint：`multi_region_description/checkpoints/qwen_person_2to5_loc_gpu2_ep2_bs16_lr8e6/final`

训练 loss：

| Epoch | Average loss |
|---|---:|
| 1 | 0.438391 |
| 2 | 0.212972 |

修正后的评估结果：

| 指标 | 结果 |
|---|---:|
| Count match rate | 1.0000 |
| Loc echo accuracy | 1.0000 |
| Positional word F1 | 0.8402 |
| Bbox-matched word F1 | 0.8402 |
| Exact match rate | 0.0325 |

按人物数量统计的 positional word F1：

| 人物数 | 样本数 | Average F1 |
|---:|---:|---:|
| 2 | 421 | 0.8547 |
| 3 | 205 | 0.8361 |
| 4 | 134 | 0.8312 |
| 5 | 72 | 0.8271 |

随着人物数量增加，描述质量缓慢下降，主要原因是区域之间的视觉串扰、相似人物混淆和描述错配，而不是输出数量不足。

## 2. 评估脚本问题说明

最初评估得到的 count match rate 为 32.57%，但该结果不是模型真实性能，而是评估脚本错误切分生成结果导致的。

Florence-2 是 encoder-decoder 模型。对于 encoder-decoder 模型，`model.generate()` 返回的是 decoder 输出，不包含 encoder prompt。然而旧评估逻辑又执行了：

```python
generated_ids[:, inputs["input_ids"].shape[1]:]
```

这会从真实生成结果开头额外删除一段 token，通常恰好删除第一个区域的描述及其部分 loc token。旧结果因此呈现出非常规律的假象：

- N=2 经常被解析为 1 个区域；
- N=3 经常被解析为 2 个区域；
- N=4 基本被解析为 3 个区域；
- N=5 基本被解析为 4 个区域；
- 560/832 个样本恰好少解析一个区域；
- 解析结果往往是 GT 的第 2 个区域到最后一个区域。

修复后，832 条测试数据在 N=2、3、4、5 上的数量匹配率均为 100%，loc token 回显准确率也为 100%。

已修复文件：

- `multi_region_description/eval/scripts/eval_plan_b.py`
- `multi_region_description/web_demo/server.py`

因此，后续 badcase 分析应使用修正后的结果：

```text
multi_region_description/eval/results/
qwen_person_2to5_loc_gpu2_ep2_bs16_lr8e6_fixed.jsonl
```

## 3. 真实 Badcase 类型

### 3.1 相邻人物描述互换

模型能够生成图中两个人物的大致正确描述，但将两个描述绑定到了相反的 bbox。

典型情况：

- bbox 1 对应白衬衫、黑外套、拿文件夹的人；
- bbox 2 对应黄衣、深色裤子、拿手机的人；
- 模型为 bbox 1 输出黄衣人物描述；
- 模型为 bbox 2 输出白衬衫人物描述。

此时每条描述单独看可能很准确，但 positional F1 很低。某个典型样本的 positional F1 为 0.543，允许交换两个预测描述后可达到 0.882。

说明模型学会了识别整图中的人物属性，但没有稳定学会 bbox 与 description 的一一绑定关系。

### 3.2 多人场景中的位置串扰

在 N=3～5 的场景中，一个 bbox 对应的描述可能吸收其他 bbox 中人物的属性、动作或位置关系。

典型表现：

- 当前 bbox 是坐在电脑前的粉色上衣人物，模型却输出附近行走男子的描述；
- 后续 bbox 又重复前面人物的描述；
- 描述整体属于图中真实人物，但人物与 bbox 的配对关系错误；
- 人物越多，平均 F1 越低。

将预测描述与 GT 描述进行最优重排后：

- 原始 positional F1 约为 0.8439；
- 最优重排 F1 约为 0.8532；
- 49 个样本重排后提升超过 0.05；
- 30 个样本提升超过 0.10；
- 8 个样本提升超过 0.20。

这说明明显的描述错位确实存在，但不是全部文本误差的唯一来源。

### 3.3 重复人物和相似外观混淆

当同一帧中多个人物具有相似外观或动作时，模型容易重复生成同一个人物模板。

高风险场景包括：

- 多人都穿黑色或深色上衣；
- 多人都坐在办公桌前；
- 多人都在使用电脑或手机；
- 多人 bbox 尺寸相近且距离较近；
- 多人姿态、朝向和背景高度相似。

典型错误是连续多次输出：

```text
A man in a black shirt sits at a desk using a computer
```

从而覆盖白衣行走人物、举手人物或其他不同人物的描述。

### 3.4 衣着颜色错误

颜色属性是最常见的局部错误之一。粗略词汇分析中，大量区域的预测颜色集合与 GT 不一致。

常见混淆：

- gray 与 black；
- white 与 light-colored；
- beige 与 gray；
- yellow shirt 与 yellow hat；
- blue 与 dark-colored；
- pink 与 red。

部分差异属于合理近义表达，但也存在明显将邻近人物颜色复制到当前人物的情况。

### 3.5 服装类别错误

常见服装类别混淆包括：

- jacket 与 hoodie；
- pants 与 jeans；
- shirt 与 top；
- coat 与 jacket；
- shorts 与 light pants；
- plaid dress 与 checkered dress。

这些错误一部分来自低分辨率或遮挡，另一部分来自相邻人物串扰。

### 3.6 动作和手持物错误

常见动作混淆：

- standing 与 walking；
- holding handlebars 与 riding a scooter；
- looking at a phone 与 using a phone；
- sitting 与 standing；
- raising an arm 与 walking；
- lying on the floor 与 sitting on the floor。

常见物体混淆：

- phone 与 laptop；
- mop 与 broom；
- dog 与 cat；
- folder 与 phone；
- bag 与其他手持物。

其中 `looking at a phone` 与 `using a phone` 等差异语义接近，但将 phone 识别为 laptop、将 dog 识别为 cat 则属于明显视觉错误。

### 3.7 场景描述过度泛化或细节缺失

模型有时能够正确描述人物主体，但会删除 GT 中的背景或关系细节。

例如：

```text
GT:   stands in a hallway with a large screen and chairs in the background
Pred: stands in a hallway
```

不过输出过短并不是主要问题：

- 预测明显短于 GT 的区域数量较少；
- 预测明显长于 GT 的区域也不多；
- 实际生成最长约 106 tokens，没有触及 `max_new_tokens=160`；
- 当前 badcase 不是生成长度截断造成的。

## 4. 根因判断

### 4.1 当前格式更强调“事后回显 loc”，而不是“先指定区域再描述”

当前 label 格式为：

```text
description1<loc_bbox1>description2<loc_bbox2>...
```

模型在生成 `description1` 时，对应 bbox 只存在于 encoder prompt 中。直到描述已经生成完毕，decoder 才输出该描述对应的 loc token。

因此 loc token 更像对输入 bbox 的事后复制，而不是生成当前描述前的显式区域条件。模型可以学会：

1. 从整图识别若干人物；
2. 按某种显著性或内部顺序生成描述；
3. 正确复制 prompt 中的 loc token；

但仍可能无法保证每条描述真正绑定到当前 bbox。

### 4.2 固定 region 顺序可能形成训练捷径

当前训练数据按照 `crop_index` 固定排序。模型可能部分学习以下捷径：

- 第一个输出描述图中最显眼人物；
- 后续描述按照视觉显著性、空间顺序或常见人物顺序生成；
- bbox token 只负责满足输出格式，并未成为主要视觉 grounding 条件。

固定顺序无法强迫模型证明自己真正理解 bbox-description 对应关系。

### 4.3 标准 Florence-2 缺乏独立 ROI 视觉特征

当前模型输入是整图和文本 loc token。模型并没有为每个 bbox 单独执行 ROIAlign，也没有显式提供每个区域的独立 crop feature。

在密集多人场景中：

- 多个人物共享相同全局视觉特征；
- bbox 只通过文本 loc token影响 cross-attention；
- 邻近人物、相似外观和重叠区域更容易发生串扰。

### 4.4 固定 5% 外扩可能增加相邻人物重叠

对于独立人物，bbox 外扩有助于补充上下文；但对于相邻人物，固定向四边外扩 5% 可能把另一个人物包含进当前区域。

风险包括：

- 两个相邻框外扩后同时包含双方身体；
- 小人物 bbox 外扩后被附近大人物视觉特征主导；
- 模型更难判断 bbox 中的中心主体；
- 相邻人物的颜色、动作和手持物发生交叉污染。

### 4.5 数据中相似描述和重复人物会放大混淆

多人办公室、展会、猫咖等场景中，经常存在：

- 多个人穿相似衣服；
- 多个人执行相似动作；
- 多个 GT description 文本高度相似；
- bbox 较小或人物被遮挡。

这种数据会使模型倾向输出高频模板，而不是精确区分局部属性。

## 5. 建议解决方案

### 5.1 优先改为 loc-first 输出格式

建议新增输出格式：

```text
<loc_bbox1>description1<sep><loc_bbox2>description2<sep>...
```

示例：

```text
<loc_879><loc_323><loc_955><loc_730>A woman in a yellow top and white pants...
<sep>
<loc_364><loc_97><loc_436><loc_341>A woman wearing a white shirt...
```

优势：

- decoder 在生成 description 前先生成对应 bbox；
- 当前描述紧邻当前 bbox token；
- bbox 能够作为当前文本段的显式前缀条件；
- 解析语法更直接：`bbox → description`；
- 比 description-first 更有利于区域与文本绑定。

这是成本最低且最值得优先验证的结构改动。

### 5.2 联合随机打乱 prompt 和 label 中的 region 顺序

训练时对同一帧的 region 顺序进行随机 permutation，并对 prompt 与 label 使用同一个 permutation。

例如原顺序：

```text
bbox1, bbox2, bbox3
```

可以扩增为：

```text
bbox3, bbox1, bbox2
bbox2, bbox3, bbox1
```

建议：

- N=2：保留原顺序和交换顺序；
- N=3：随机生成 2～3 个顺序；
- N=4/5：随机生成 2 个顺序；
- 测试集保持固定顺序；
- 不需要穷举所有排列。

随机顺序可以破坏“按视觉显著性或固定位置输出”的捷径，迫使模型学习 bbox-description 对应关系。

### 5.3 混入单区域辅助训练样本

从多区域数据中展开单区域样本：

```text
Prompt: <REGIONS_TO_DESCRIPTIONS><loc_bbox2>
Label:  <loc_bbox2>description2
```

建议训练采样比例：

- 60% 多区域样本；
- 25% 单区域样本；
- 15% 相邻人物 hard-negative 样本。

单区域数据可以强化最基础的能力：

```text
给定一个 bbox，只描述这个 bbox 中的人物
```

也可以混入原 checkpoint 的 `REGION_TO_DESCRIPTION` 训练数据，降低多区域微调对原始描述能力的遗忘。

### 5.4 构造相邻人物 hard cases

可以根据 bbox 自动筛选困难样本：

- bbox 中心距离较小；
- bbox IoU 较高；
- 一个 bbox 大量包含另一个 bbox；
- 两个人物服装颜色明显不同；
- 两个人物动作或手持物不同；
- 描述文本相似但关键属性不同。

训练时提高这些样本的采样权重：

- 普通样本采样 1 次；
- 困难样本采样 2～3 次；
- 同时加入 region 顺序交换版本。

最有效的 hard case 是“人物相邻但属性明显不同”的两人样本，例如：

```text
bbox A: white shirt, holding folder
bbox B: yellow shirt, holding phone
```

### 5.5 使用自适应 bbox 外扩

不建议对所有人物统一使用固定 5% 外扩。可以根据与其他人物框的关系动态调整：

- 独立人物：保持 5% 外扩；
- 相邻人物：缩小到 0～2%；
- 外扩后与其他人物 IoU 明显上升：回退到原 bbox；
- 只向没有其他人物的方向扩展；
- 对小框可适当增加上下文，但限制不能包含其他人物中心点。

建议对比三套验证数据：

1. 原始 bbox；
2. 固定 5% 外扩 bbox；
3. 避免人物重叠的自适应外扩 bbox。

### 5.6 显式增强区域视觉信息

如果仅调整语法和数据仍不足，可以考虑让模型获得更直接的区域视觉信息。

低成本方案：

- 在整图中绘制细 bbox；
- 不遮挡人物主体；
- 每个 bbox 使用固定颜色或编号；
- prompt 和输出中加入对应编号。

更强但改动更大的方案是构造 contact sheet：

- 左侧保留整图；
- 右侧按顺序排列 crop #1、crop #2、crop #3；
- 模型同时看到全局场景和每个人物的放大 crop。

该方案能显著减少小人物和相邻人物串扰，但需要重新设计图像构造与训练输入。

### 5.7 不建议仅增加 epoch

当前模型已经达到：

- count match 100%；
- loc echo 100%；
- epoch 2 loss 约 0.213。

说明格式和 loc 复制已经充分学会。继续在相同数据上增加 epoch 更可能导致：

- 进一步记忆描述模板；
- 重复描述倾向增强；
- 对原 checkpoint 能力产生遗忘；
- bbox grounding 不一定改善。

下一轮应优先改变数据组织和输出语法，而不是简单把 epoch 增加到 3～5。

建议下一轮超参数：

| 参数 | 建议值 |
|---|---:|
| Epochs | 2 |
| Batch size | 16 |
| Learning rate | `5e-6`～`6e-6` |
| Warmup ratio | 约 5% |
| Precision | BF16 |
| Max grad norm | 1.0 |
| Target max length | 根据 loc-first 数据重新统计，预计 160 足够 |

### 5.8 增加 grounding 专用评估指标

训练 loss 和 count match 无法反映描述是否绑定到了正确 bbox。后续评估至少应包含：

- Count match rate；
- Loc echo accuracy；
- Positional word F1；
- 最优 permutation word F1；
- Assignment gap；
- 颜色属性准确率；
- 服装类别准确率；
- 动作准确率；
- 手持物准确率；
- 相邻人物子集 F1；
- 高 bbox-overlap 子集 F1。

定义：

```text
assignment_gap = best_permutation_f1 - positional_f1
```

如果 `assignment_gap` 较高，说明模型识别到了图中人物，但将描述分配给了错误 bbox。该指标比单纯 positional F1 更适合衡量人物互换问题。

## 6. 推荐实验顺序

建议按单变量或少变量方式逐步验证，不要一次修改全部因素。

### 实验 1：loc-first

- 将 label 改为 `<loc>description<sep>`；
- 其他数据和超参数保持不变；
- 与当前 description-first 模型直接比较。

### 实验 2：loc-first + region permutation

- 在实验 1 基础上随机打乱 region 顺序；
- prompt 和 label 使用相同 permutation；
- 重点观察 assignment gap 和相邻人物子集 F1。

### 实验 3：加入单区域辅助数据

- 使用约 60% 多区域、25% 单区域、15% hard cases；
- 降低 learning rate 到 `5e-6`～`6e-6`；
- 保持 2 epochs。

### 实验 4：相邻人物 hard-case 过采样

- 根据 bbox 距离、IoU 和属性差异筛选困难样本；
- 困难样本提高到普通样本 2～3 倍采样权重；
- 单独报告 hard subset 指标。

### 实验 5：自适应 bbox 外扩

- 比较原框、固定 5% 外扩和自适应外扩；
- 重点检查相邻人物、重叠人物和小人物。

### 实验 6：显式区域视觉输入

仅当上述低成本方案仍无法满足要求时，再尝试：

- 绘制彩色 bbox；
- bbox 编号；
- 整图 + crop contact sheet；
- 更深层的 ROI feature 或模型结构修改。

## 7. 工程上的可靠 fallback

如果业务优先保证准确率，而不是单次推理吞吐，最可靠方案仍然是：

```text
每个 bbox 单独执行一次 REGION_TO_DESCRIPTION
```

优点：

- 每次只有一个目标区域；
- 不存在区域间输出顺序问题；
- 相邻人物串扰明显减少；
- 可以直接复用基础 checkpoint 的单区域能力。

缺点：

- 每张图需要执行 N 次生成；
- 图像 encoder 可能重复计算；
- 推理延迟和吞吐低于单次多区域生成。

可以采用混合策略：

1. 默认使用多区域单次生成；
2. 对 bbox 高重叠、人物距离近或模型置信度低的样本触发单区域 fallback；
3. 仅对疑似互换的区域重新推理，而不是整张图全部重跑。

## 8. 当前结论

当前模型已经可靠学会多区域输出语法、区域数量控制和 loc token 回显。下一阶段的核心目标不是继续强化格式，而是增强：

```text
bbox → 对应人物视觉特征 → description
```

优先级最高的两个改动是：

1. 使用 loc-first 输出格式；
2. 对 region 顺序进行联合随机打乱。

随后建议加入单区域辅助数据和相邻人物 hard-case 过采样。如果这些数据与语法层面的改动仍不足，再考虑显式 crop 视觉输入或单区域推理 fallback。
