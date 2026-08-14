# 人物属性描述 RL 测试评估报告

测试集：`data/prepared/test.jsonl`，4,328 个样本
基线：`v4b_sft`
Bootstrap：session 级重采样，2,000 次，seed 20260720
生成日期：2026-08-12

| 候选 | Checkpoint |
|---|---|
| `v4b_sft` | `artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final` |
| `qwen_rl` | `artifacts/rl/region_category_person_scst_reviewed5981_qwen4b/final` |
| `lexical_rl` | `artifacts/rl/region_category_person_scst_reviewed5981_lexical/final` |

---

## 0. 评估修正说明（2026-08-12）

原评估脚本的 `normalize_value()` 对列表型 GT 属性（如 `['brown']`、`['black','white']` 双色衣物）返回 `None`，导致该字段 `gt_positive=0`；模型只要作答就被记为 fabrication（幻觉），**即使语义完全正确**。测试集中受影响字段共 11,648 个（10,173 个单元素列表 + 1,475 个多元素列表），涉及 2,912 / 4,328 行（67.3%），集中在 `upper_garment.color`、`lower_garment.color`、`shoes.color`、`head.accessories` 四个字段。

修正措施：
1. GT 源切换至 `data/prepared/test_qwen_attributes.jsonl`（从人工修正属性改写后的 GT caption 反向抽取，天然为干净字符串）
2. 新增 `normalize_value_set()`，打分时对列表 GT 取最佳匹配元素，作为兜底保险

修正效果（三个候选一致）：lexical F1 +4.1～4.5pp，幻觉率 22.2～22.6% → 10.6～11.1%。**相对排名与推荐结论不变**。原冻结结果已快照至 `artifacts/rl/evaluation_frozen_20260812/`。

> 训练侧未受影响：`rl.jsonl` 与 `dev.jsonl` 中列表型 GT 数量为 **0**，RL 训练奖励信号从未接触过这批数据，因此无需重跑 RL。

---

## 1. 指标定义

### 1.1 打分基础：软 F1（soft F1）

逐字段将 GT 与模型抽取属性配对，用相似度加权计数，而非硬性 0/1 判定：

| 计数项 | 含义 |
|---|---|
| `gt_positive` | GT 该字段有明确值（非 unknown）的样本数，即"应该答对"的题量 |
| `n_gen_assert` | 模型给出具体值的次数，即"作答"总量 |
| `soft_tp` | 相似度累加值 `Σs`，s ∈ [0,1] |
| `soft_fp` | `Σ(1−s)`，作答但不完全正确的部分 |
| `soft_fn` | `Σ(1−s)` + GT 有值而模型未提及的次数 |
| `n_fab` | GT 确为 unknown 但模型作答的次数（幻觉） |

由此 `Precision = soft_tp/(soft_tp+soft_fp)`，`Recall = soft_tp/(soft_tp+soft_fn)`，`F1 = 2PR/(P+R)`。

### 1.2 核心指标

| 指标 | 定义 | 方向 |
|---|---|---|
| **Lexical Micro P/R/F1** | 相似度用 token 重叠（`value_similarity`）计算，全部字段汇总后算 micro 平均。严格、可复现、无模型依赖 | 越高越好 |
| **Qwen Semantic Micro P/R/F1** | 相似度改用 Qwen3.5-4B 判分器 5 级评分 {1.0, 0.75, 0.5, 0.25, 0.0}，能识别"navy"≈"dark blue"这类同义表述。更接近人类判断 | 越高越好 |
| **Joint Attribute F1** | `0.5 × (Lexical F1 + Semantic F1)`，**主决策指标**，兼顾严格匹配与语义等价 | 越高越好 |
| **Lexical Macro-field F1** | 各字段 F1 的算术平均（不按样本量加权）。低频字段权重被放大，反映"冷门属性"表现 | 越高越好 |
| **Fabrication ratio** | `n_fab / n_gen_assert`。F1 的盲区补充：F1 只惩罚答错，不惩罚"GT 无此项而模型凭空编造" | 越低越好 |

### 1.3 质量门槛指标

| 指标 | 定义 | 门槛 |
|---|---|---|
| **Structure pass rate** | caption 为单句且以主语开头的比例 | 绝对值 ≥ 95%，相对基线不得下降 > 1pp |
| **Background leakage rate** | caption 提及场景/建筑/地点/背景的比例（应只描述人物本身） | 绝对值 ≤ 1% |
| **Invalid output rate** | 空输出、JSON 残留、管道分隔字段等格式崩坏比例 | 相对基线不得上升 > 0.5pp |

### 1.4 参考性指标（不参与选择判定）

| 指标 | 定义 |
|---|---|
| **Length mean** | 平均词数 |
| **18-24 ratio** | 落在 18–24 词理想区间的比例 |

> 长度指标仅作参考记录，**不作为决定性选择指标**。有些人物可见属性本就很多、有些很少，符合长度区间的样本多当然更好，但不具决定性。

### 1.5 统计方法

- **配对差值**：同一批样本、同一套 GT、同一抽取器，三个候选唯一差异是 Florence checkpoint，因此可做严格配对比较
- **95% 置信区间**：以 **session** 为重采样单位（同一 session 内样本高度相关，按样本重采样会低估方差），2,000 次 bootstrap，取差值分布的 2.5 / 97.5 分位数
- **显著性判据**：CI 完全排除 0 即为统计显著
- **预注册推荐门槛**：joint F1 提升 **> 0.5pp** 且 CI 下界 > 0（冻结协议事前设定，不允许事后放宽）

---

## 2. 核心指标对比

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
| *18-24 ratio（参考）* | *49.12%* | *52.56%* | *53.17%* | — |

**读法**：lexical_rl 在 9 项核心指标中 8 项领先，唯一落后的是 Qwen semantic precision（-0.146pp）。qwen_rl 除幻觉率最低外，其余核心指标全面弱于基线。

---

## 3. 相对 V4B 的配对差值与 95% 置信区间

### 3.1 qwen_rl vs v4b_sft

| 指标 | 差值 | 95% CI | 显著性 |
|---|---:|---:|:---:|
| **Joint Attribute F1** | **−0.156pp** | [−0.277, −0.034] | ❌ 显著变差 |
| Lexical Micro F1 | −0.219pp | [−0.356, −0.079] | ❌ 显著变差 |
| — Precision | −0.219pp | [−0.355, −0.085] | ❌ 显著变差 |
| — Recall | −0.217pp | [−0.387, −0.044] | ❌ 显著变差 |
| Qwen Semantic F1 | −0.094pp | [−0.214, +0.032] | ➖ 不显著 |
| — Precision | −0.092pp | [−0.207, +0.029] | ➖ 不显著 |
| — Recall | −0.095pp | [−0.254, +0.078] | ➖ 不显著 |
| Lexical Macro-field F1 | −0.288pp | [−0.893, +0.591] | ➖ 不显著 |
| **Fabrication ratio** | **−0.440pp** | [−0.597, −0.276] | ✅ 显著改善 |
| Structure pass rate | ±0.000pp | [0.000, 0.000] | ➖ 无变化 |
| Background leakage | ±0.000pp | [0.000, 0.000] | ➖ 无变化 |
| Invalid output | ±0.000pp | [0.000, 0.000] | ➖ 无变化 |
| *Length mean（参考）* | *−0.55 词* | *[−0.63, −0.48]* | — |
| *18-24 ratio（参考）* | *+3.44pp* | *[+2.45, +4.45]* | — |

**门槛判定**：全部通过（`eligible: true`）。但 joint F1 显著为负，不具备推荐资格。

### 3.2 lexical_rl vs v4b_sft

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
| Background leakage | ±0.000pp | [0.000, 0.000] | ➖ 无变化 |
| Invalid output | ±0.000pp | [0.000, 0.000] | ➖ 无变化 |
| *Length mean（参考）* | *−0.48 词* | *[−0.54, −0.40]* | — |
| *18-24 ratio（参考）* | *+4.04pp* | *[+3.05, +4.98]* | — |

**门槛判定**：全部通过（`eligible: true`）。joint F1 显著改善且 CI 排除 0，但 **+0.207pp 未达预注册的 0.5pp 门槛**。

---

## 4. 逐样本胜负分析

按每个样本的 F1 与基线逐一比较（4,328 个样本）：

| 候选 | 路线 | 胜 | 平 | 负 | 净胜 | 胜率（不含平） |
|---|---|---:|---:|---:|---:|---:|
| qwen_rl | 词法 | 507 | 3,220 | 601 | **−94** | 45.8% |
| qwen_rl | Qwen 语义 | 485 | 3,321 | 522 | **−37** | 48.2% |
| lexical_rl | 词法 | 730 | 3,019 | 579 | **+151** | **55.8%** |
| lexical_rl | Qwen 语义 | 577 | 3,216 | 535 | **+42** | **51.9%** |

**读法**：
- lexical_rl 在两条评分路线上均为净胜，词法路线优势明显（+151，胜率 55.8%）
- qwen_rl 在两条路线上均为净负，与其 joint F1 显著变差一致
- 平局占比高（70–77%）说明 RL 主要改动少数样本，未大范围重写风格 —— 这符合 SCST 保守微调的预期

---

## 5. 逐字段分析

### 5.1 基线各字段表现与样本量

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

**基线短板**：`handheld_items.dangerous_item`（3.70%，仅 25 样本）、`handheld_items.mobile_phone`（18.03%）、`extra`（26.88%）、`carried_items.handbag`（31.72%）、`head.hairstyle`（42.07%）。手持/携带物品与发型是整体最弱环节。

### 5.2 lexical_rl 逐字段差值

| 字段 | F1 差值 | soft_tp Δ | soft_fp Δ | soft_fn Δ | 评价 |
|---|---:|---:|---:|---:|---|
| handheld_items.dangerous_item | **+7.011pp** | +1.0 | +0.0 | −1.0 | ⚠️ 仅 25 样本，噪声大 |
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

**净效应**：11 项改善、8 项退化。改善集中在**高频可见属性**（hair_color 3,679 样本、shoes.type 2,201、lower_garment.length 2,617、两个 color 字段各 3,600+），退化集中在**低频主观属性**（hairstyle 1,096、extra 267、handbag 325、backpack 277）。从加权贡献看这是正向交换 —— 这也解释了为何 micro F1 显著改善而 macro F1 改善不显著（macro 放大了低频字段的退化）。

### 5.3 qwen_rl 逐字段差值

| 字段 | F1 差值 | soft_tp Δ | soft_fp Δ | soft_fn Δ | 评价 |
|---|---:|---:|---:|---:|---|
| handheld_items.dangerous_item | **+7.011pp** | +1.0 | +0.0 | −1.0 | ⚠️ 仅 25 样本，噪声大 |
| extra | **−3.922pp** | −7.3 | −1.7 | +7.3 | ❌ 主要退化 |
| carried_items.backpack | −1.971pp | −6.9 | −4.1 | +6.9 | ❌ 退化 |
| head.hairstyle | −1.896pp | −17.0 | +15.0 | +17.0 | ❌ 退化 |
| handheld_items.mobile_phone | −1.244pp | −7.1 | −12.9 | +7.1 | ➖ 精度升召回降 |
| shoes.type | **+1.171pp** | +26.9 | −12.9 | −26.9 | ✅ 改善 |
| carried_items.handbag | −1.023pp | −3.8 | −4.2 | +3.8 | ❌ 退化 |
| head.accessories | −0.963pp | −10.9 | +3.9 | +10.9 | ❌ 退化 |
| upper_garment.color | −0.718pp | −29.0 | +34.0 | +29.0 | ❌ 高频字段退化 |
| lower_garment.color | −0.647pp | −22.1 | +25.1 | +22.1 | ❌ 高频字段退化 |
| lower_garment.length | −0.566pp | −12.0 | +13.0 | +12.0 | ❌ 退化 |
| shoes.color | −0.380pp | −4.2 | +15.2 | +4.2 | ❌ 退化 |
| age_group | +0.314pp | +17.6 | −6.6 | −17.6 | ✅ 改善 |
| upper_garment.type | −0.308pp | −11.9 | +15.9 | +11.9 | ➖ 轻微退化 |
| head.hair_color | −0.244pp | −10.1 | +5.1 | +10.1 | ➖ 轻微退化 |
| upper_garment.length | −0.176pp | −11.5 | +1.5 | +11.5 | ➖ 轻微退化 |
| gender | +0.168pp | +12.6 | +0.4 | −12.6 | ✅ 改善 |
| head.hair_length | −0.086pp | −4.0 | +1.0 | +4.0 | ➖ 轻微退化 |
| lower_garment.type | +0.011pp | +0.4 | −0.4 | −0.4 | ➖ 持平 |

**净效应**：仅 4 项改善、15 项退化，且退化涵盖 `upper_garment.color`、`lower_garment.color` 等最高频字段。与 lexical_rl 形成鲜明对比：Qwen 语义奖励未能转化为测试集上的属性准确率提升。

### 5.4 两条 RL 路线的共同特征

| 现象 | qwen_rl | lexical_rl | 解释 |
|---|---|---|---|
| `extra` 字段退化 | −3.922pp | −3.300pp | 两者都倾向于输出更精简的 caption，牺牲了自由描述项的覆盖 |
| `carried_items.*` 退化 | −1.97 / −1.02pp | −1.03 / −1.54pp | 携带物品是低频且易漏项，RL 收紧输出后漏得更多 |
| `head.hairstyle` 退化 | −1.896pp | −3.417pp | 发型描述主观性强，奖励函数难以给出稳定信号 |
| 幻觉率下降 | −0.440pp | −0.267pp | 奖励函数中 −0.40 权重的 fabrication 惩罚项确实生效 |
| 长度趋短 + 合规率上升 | −0.55 词 / +3.44pp | −0.48 词 / +4.04pp | 奖励函数中的长度项（理想 18–24 词）生效 |

**共性结论**：SCST 奖励函数在"减少幻觉"和"控制长度"两个显式目标上对两条路线均有效。差异在属性准确率：词法奖励与测试指标同源，因此 lexical_rl 的改善能直接体现；Qwen 语义奖励与测试指标存在口径差异，qwen_rl 的优化方向未能对齐测试集表现。

---

## 6. 结论

### 6.1 学术推荐（严格按冻结协议）

**推荐：`v4b_sft`（基线，即不采用 RL checkpoint）**

判定链条：

| 步骤 | qwen_rl | lexical_rl |
|---|---|---|
| 1. 质量门槛 | ✅ 全部通过 | ✅ 全部通过 |
| 2. joint F1 差值方向 | ❌ −0.156pp（显著为负） | ✅ +0.207pp（显著为正） |
| 3. CI 下界 > 0 | ❌ [−0.277, −0.034] | ✅ [+0.070, +0.347] |
| 4. 提升 > 0.5pp 预注册门槛 | ❌ | ❌ **+0.207pp < 0.5pp** |
| **结论** | 不合格 | **卡在第 4 步** |

`lexical_rl` 的改善**统计显著但幅度不足**。预注册的 0.5pp 门槛是在评估前冻结设定的，其作用正是防止事后为了让某个候选通过而放宽标准。因此严格按协议，学术结论为维持基线。

这个结论的实质含义：**本轮 SCST 未能带来足以支撑发布决策的属性准确率提升**，而非"RL 无效"。

### 6.2 业务推荐

**推荐：`lexical_rl`**

若决策目标是"当前可交付的最佳模型"而非"是否通过预注册假设检验"，证据支持采用 lexical_rl：

| 维度 | 证据 | 判断 |
|---|---|---|
| 属性准确率 | joint F1 +0.207pp，CI [+0.070, +0.347] 完全排除 0 | ✅ 真实改善，非噪声 |
| 词法路线 | F1 +0.362pp，精度召回双升 | ✅ 一致改善 |
| 幻觉率 | −0.267pp，CI [−0.427, −0.109] | ✅ 显著降低，对业务可信度直接有利 |
| 逐样本 | 词法净胜 +151（55.8%），语义净胜 +42（51.9%） | ✅ 两路线一致净胜 |
| 高频属性 | hair_color +1.86pp、shoes.type +1.82pp、lower_garment.length +1.21pp | ✅ 用户最常关注的可见属性改善 |
| 输出简洁度 | 平均 −0.48 词，18-24 词合规率 +4.04pp | ✅ 参考项，利于下游消费 |
| 格式稳定性 | 结构合规 98.198%、背景泄漏 0.069%、无效输出 0% | ✅ 与基线持平 |

**需要接受的代价**：

| 风险项 | 数值 | 影响评估 |
|---|---|---|
| Qwen 语义精度 | −0.146pp，CI [−0.280, −0.012] | ⚠️ 轻微显著下降。语义召回 +0.211pp 有余量补偿，语义 F1 净 +0.052pp 仍为正 |
| `head.hairstyle` | −3.417pp（1,096 样本） | 若业务需要发型描述，此项退化需评估 |
| `extra` 自由属性 | −3.300pp（267 样本） | caption 更精简的直接代价 |
| `carried_items.handbag` | −1.540pp（325 样本） | 若业务关注携带物品检出，需评估 |

**决策建议**：若业务核心是**衣着颜色/类型、发色、鞋类等高频可见属性**，lexical_rl 是明确更优选择，同时获得更低幻觉率与更简洁输出。若业务强依赖**发型描述或携带物品检出**，则 `head.hairstyle` −3.42pp 与 `carried_items.handbag` −1.54pp 的退化需要单独权衡，此时维持基线更稳妥。

### 6.3 `qwen_rl` 明确不推荐

joint F1 −0.156pp，CI [−0.277, −0.034] 完全落在负区间；逐样本两条路线均净负；15 / 19 个字段退化，且涵盖 `upper_garment.color`、`lower_garment.color` 等最高频字段。唯一优势是幻觉率最低（10.633%，比 lexical_rl 低 0.173pp），但不足以补偿属性准确率的系统性下降。

技术归因：Qwen3.5-4B 判分器的 5 级语义奖励与测试集的评分口径存在系统性差异，导致训练优化方向与评测目标未对齐。相比之下 lexical 奖励与测试指标同源（均为 `value_similarity` token 重叠），优化信号直接可迁移。

### 6.4 后续建议

1. **若要再跑一轮 RL**：优先改进奖励函数对 `head.hairstyle`、`extra`、`carried_items.*` 的处理 —— 当前奖励函数的长度惩罚项可能过度压制了这些低频描述项。可考虑对低频字段的召回单独加权。
2. **`handheld_items.dangerous_item` 需扩充样本**：仅 25 个 GT 正样本，F1 3.70%，任何差值（本轮两个候选均为 +7.011pp）都是噪声，不具统计意义。
3. **qwen 语义奖励路线若要保留**：需先校准判分器与评测口径的一致性，否则训练信号与评测目标持续错位。
4. **COCO 通用能力评估尚未执行**：这是发布决策的剩余空缺项，用于确认 RL 未损害模型的通用描述能力。
