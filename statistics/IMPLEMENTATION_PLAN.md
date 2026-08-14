# 实现计划：人体属性像素占比 vs 检出 Precision 关系分析

## 目标

用 checkpoint `iter_22000` 在 val 集（7927 张图）上推理，分析 8 个人体属性（background, upper, lower, dress, hat, glasses, mask, hair）的 **GT/Pred 像素占比** 与 **检出 Precision** 的关系，结果输出为一个独立 HTML 文件。

## 输入

| 项目 | 路径 |
|------|------|
| 配置 | `segment-color/experiment/PaddleSeg/experiments/clothes_color_rgb_only/configs/optC_scratch_rgb.yml` |
| 检查点 | `segment-color/experiment/PaddleSeg/experiments/pp_liteseg_stdc2_clothes_color_2head_rgb_3ch_256_optC_scratch/checkpoints/iter_22000/model.pdparams` |
| 验证集 | `segment-color/data/splits/val.txt`（7927 行） |
| 数据集类 | `ClothesColorRGBDataset`（`clothes_color_rgb_only/dataset_rgb.py`） |
| 模型类 | `PPLiteSegWithColorHeads`（`clothes_color_poseprior_stdc2/clothes_color_model.py`） |

## 输出

所有输出放在 `segment-color/statistics/` 下：

```
segment-color/statistics/
├── IMPLEMENTATION_PLAN.md              # 本文件
├── predict_detection.py                # 脚本1：推理，输出 CSV
├── per_image_detection.csv             # 中间产物：7927 行逐图检出数据
└── detection_analysis_report.html      # 最终产物：独立 HTML 文件
```

## 关键定义

### 检出（Detection）

对一张图的某个属性：
- **GT present**：GT 标注 mask 中该类像素数 > 0 → 1，否则 0
- **Pred present**：模型预测 mask 中该类像素数 > 0 → 1，否则 0
- **TP**：GT present=1 且 Pred present=1
- **FP**：GT present=0 但 Pred present=1
- **FN**：GT present=1 但 Pred present=0
- **TN**：GT present=0 且 Pred present=0
- **Precision**：TP / (TP + FP)（全局聚合）
- **Recall**：TP / (TP + FN)（全局聚合）
- **F1**：2 * P * R / (P + R)

### 像素占比（Pixel Ratio）

- **GT ratio**：GT 中该类像素数 / 整图像素总数（256×256 = 65536）
- **Pred ratio**：Pred 中该类像素数 / 整图像素总数

---

## 脚本 1：`predict_detection.py`

### 功能

遍历 val 集 7927 张图，逐图推理，输出 `per_image_detection.csv`。

### 依赖

- PaddleSeg 框架（`Config`, `SegBuilder`）
- `ClothesColorRGBDataset`（RGB-only 数据集）
- `PPLiteSegWithColorHeads`（模型）
- 需要将 `clothes_color_poseprior_stdc2` 和 `clothes_color_rgb_only` 加入 sys.path

### CSV 字段

```
image_rel
background_gt_present, background_pred_present, background_gt_ratio, background_pred_ratio
upper_gt_present, upper_pred_present, upper_gt_ratio, upper_pred_ratio
lower_gt_present, lower_pred_present, lower_gt_ratio, lower_pred_ratio
dress_gt_present, dress_pred_present, dress_gt_ratio, dress_pred_ratio
hat_gt_present, hat_pred_present, hat_gt_ratio, hat_pred_ratio
glasses_gt_present, glasses_pred_present, glasses_gt_ratio, glasses_pred_ratio
mask_gt_present, mask_pred_present, mask_gt_ratio, mask_pred_ratio
hair_gt_present, hair_pred_present, hair_gt_ratio, hair_pred_ratio
mIoU, Acc
```

### 推理流程

```python
# 伪代码
cfg = Config("optC_scratch_rgb.yml")
model = SegBuilder(cfg).model
paddle_utils.load_entire_model(model, "iter_22000/model.pdparams")
model.eval()

val_dataset = ClothesColorRGBDataset(mode="val", ...)
loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

for data in loader:
    img = data["img"]           # [1, 3, 256, 256]
    seg_gt = data["seg_label"]  # [1, 1, 256, 256]

    seg_pred = model(img)[0]    # [1, 8, 256, 256] -> argmax -> [1, 256, 256]

    for c in 0..7:
        gt_mask = seg_gt == c
        pred_mask = seg_pred == c
        gt_present = gt_mask.sum() > 0
        pred_present = pred_mask.sum() > 0
        gt_ratio = gt_mask.sum() / total_pixels
        pred_ratio = pred_mask.sum() / total_pixels
        # write to CSV row
```

### 运行方式

```bash
cd /data1/work/MichaelYu/segment-color/experiment/PaddleSeg
CUDA_VISIBLE_DEVICES=0 python /data1/work/MichaelYu/segment-color/statistics/predict_detection.py
```

预计耗时：7927 张图 × ~0.05s/图 ≈ 6-7 分钟。

---

## 脚本 2：`gen_detection_html.py`（或内嵌在 HTML 生成逻辑中）

### 功能

读取 `per_image_detection.csv`，计算所有统计量，生成独立 HTML 文件。

### HTML 结构

#### （1）顶部汇总表

| 属性 | Precision | Recall | F1 | GT平均占比 | Pred平均占比 | TP | FP | FN |
|------|-----------|--------|----|-----------|-------------|----|----|----|
| background | 0.999 | ... | ... | 0.45 | 0.44 | 7000 | 10 | 5 |
| upper | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

全局 mIoU 和 Acc 也一并展示。

#### （2）占比 vs 检出 Precision 统计图

- **布局**：4 列 × 2 行，共 8 张子图
- **每张子图**：
  - X 轴：像素占比（分 10 个等宽桶）
  - Y 轴：桶内检出 Precision
  - 蓝色折线 + 散点：GT 占比 → Precision
  - 橙色折线 + 散点：Pred 占比 → Precision
  - 每个散点上标注该桶样本数
- **分桶逻辑**：对某属性，将所有图按占比排序，等宽分 10 桶（min_ratio 到 max_ratio），每桶内计算全局 Precision = ΣTP / (ΣTP + ΣFP)
- **图表库**：内嵌 Chart.js（CDN）

#### （3）Badcase 展示

- 全局 **FP 总数** 排序，取 Top-20：一张图中 8 个属性里 FP 最多的（模型多检了很多不该有的属性）
- 全局 **FN 总数** 排序，取 Top-20：一张图中 FN 最多的（模型漏检了很多该有的属性）
- 每个 badcase 显示：
  - 原始 RGB 图
  - GT 分割 mask（彩色）
  - Pred 分割 mask（彩色）
  - 标注该图具体哪些属性是 FP/FN
- 图片以 Base64 嵌入，缩略图约 256×256

### 分桶精度计算细节

```python
# 对每个属性 attr
# 收集所有图像的 (gt_ratio, pred_ratio, tp, fp, pred_present)
# GT-ratio 分桶
bins = 10
gt_bucket_stats = []  # per bucket: {total_tp, total_pred_present, precision, count}

# Pred-ratio 分桶（同理)
pred_bucket_stats = []

# 每个桶的 precision = total_tp / total_pred_present
# 如果 total_pred_present == 0，precision = NaN（不画点）
```

---

## HTML 生成实现细节

### 技术选型

- 纯静态 HTML 单文件
- **Chart.js v4** CDN 引入（`<script src="https://cdn.jsdelivr.net/npm/chart.js@4">`）
- JSON 数据内嵌在 `<script>` 标签中
- Badcase 图片用 Base64 直接嵌入 `<img src="data:image/png;base64,...">`
- CSS：简单的 grid/flex 布局，深色或浅色主题

### 生成方式

方案 A：Python 脚本直接拼接 HTML 字符串  
方案 B：Python 脚本生成 JSON 数据，搭配模板 HTML 文件

选择 **方案 A**：写一个 Python 函数逐段产出 HTML，便于控制结构和嵌入数据。

### 数据集加载（用于 badcase 图片）

Badcase 图片需要重新加载原图、GT mask、Pred mask：
- 原图：通过 `ClothesColorRGBDataset` 获取
- GT mask：从数据集获取 seg_label
- Pred mask：通过模型推理获取
- 推理时同时保存 seg_pred 到临时数组，供 badcase 使用

### 内存优化

7927 张图的 pred mask 不需要全部保存。策略：
- 第一遍推理：输出 CSV，同时把每张图的 `image_rel` + `pred_mask` 暂存到磁盘（NPY 文件）
- 或者：推理时在内存中只保留 FP/FN 较多的 badcase 候选（如每图 FP > 2 或 FN > 2），其余丢弃

选择后者（内存候选），更简洁。

---

## 参考代码

| 文件 | 用途 |
|------|------|
| `clothes_color_zeropose_ablation/predict_val_optC.py` | 推理框架参考（数据集加载、模型加载、逐图推理流程） |
| `clothes_color_zeropose_ablation/gen_color_error_html.py` | HTML 生成参考（Base64 嵌入、混淆矩阵、badcase 展示） |
| `clothes_color_rgb_only/dataset_rgb.py` | 使用的数据集类 |
| `clothes_color_poseprior_stdc2/clothes_color_model.py` | 模型类定义 |

---

## 执行顺序

1. **先执行** `predict_detection.py` → 产出 `per_image_detection.csv`
2. **再执行** `gen_detection_html.py` → 读取 CSV，产出 `detection_analysis_report.html`
3. 浏览器打开 `detection_analysis_report.html` 查看结果
