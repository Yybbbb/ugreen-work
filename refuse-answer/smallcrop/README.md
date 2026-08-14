# 小 Crop 图颜色属性预测拒绝策略 — 统计分析报告

## 1. 统计目的

在摄像头人体属性识别部署场景中，当检测到的人体目标过小（远距离、低分辨率），裁剪出的 person crop 图片像素数不足以支撑可靠的颜色属性预测。本次统计旨在：

- **量化**图片尺寸与分割/颜色分类性能之间的定量关系
- 给出**数据驱动的尺寸准入阈值**，供工程端配置「小图拒绝预测」逻辑
- 将拒绝策略从经验判断升级为**可验证、可复现的数值依据**

核心问题：**当 crop 图小到什么程度时，颜色预测的 Precision 下降到不可接受的水平？**

## 2. 数据来源

### 2.1 数据集

| 项目 | 说明 |
|------|------|
| 数据集路径 | `segment-color/data/LIP_clothes_accessory_unified/` |
| 标注总量 | 89,221 张 JSON 标注 |
| Train / Val 划分 | `splits/clothes_accessory_poseprior/train.txt` (46,878) / `val.txt` (6,391) |
| 划分策略 | val_ratio=0.12, seed=20260518, 仅保留 quality=database/reid 的图片 |
| 数据来源 | 三个子集：`lip` (LIP 公开集)、`ipc_seg_annotation` (IPC 场景标注)、`upper_lower_person_export` (额外导出) |
| 标注内容 | 分割 mask（upper/lower/dress/hat/glasses/mask/hair 共 7 个前景类）+ 11 类颜色标签 (black/white/gray/red/yellow/green/blue/purple/pink/orange/brown) |

### 2.2 模型

| 项目 | 说明 |
|------|------|
| 架构 | PP-LiteSeg + STDC2 骨干 |
| 任务 | 双头：语义分割（8 类）+ 颜色分类（上衣 11 类 + 下衣 11 类） |
| 输入 | 6 通道（RGB ×3 + Pose Heatmap ×3），短边 resize 至 256px，pad 至正方形 |
| 训练 | 从头训练 (scratch)，优化器配置 OptC |
| 检查点 | iter_40000 |
| 实验目录 | `segment-color/experiment/PaddleSeg/experiments/pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256_optC_scratch/` |
| 推理输出 | `segment-color/experiment/PaddleSeg/output/pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256_optC_scratch/pred_val_iter40000/` |
| 验证集全局指标 | mIoU=0.636, 上衣颜色 Acc=0.834, 下衣颜色 Acc=0.816 |

## 3. 统计方法

### 3.1 尺寸分析

- 读取全部 53,269 张 train+val 图片的 PIL 尺寸
- 统计宽度、高度、面积 (kpx)、宽高比、短边 (min(W,H)) 的分布
- 按面积百分位分 6 档（P0–10, P10–25, P25–50, P50–75, P75–90, P90–100）

### 3.2 性能关联分析

- 将验证集 6,391 张图片按面积排序后分档
- 每档计算：mIoU、上衣/下衣 IoU、颜色 Precision（= 正确预测数 / 总预测数）
- 每档展示上衣颜色 Precision Top-3 和对应的预测次数
- 扫描短边（32–256px）和面积（5–200 kpx）的阈值，给出单/双维度的拒绝方案

### 3.3 Bad Case 采样

- 每档按错误率抽样 8 张颜色预测错误的图片
- 展示原图缩略图（保持原始宽高比）及 GT → Pred 颜色标签

### 3.4 指标选择：为什么用 Precision？

部署时每次模型输出一个颜色预测，下游据此决策。Precision 回答「这次预测可信吗？」；Recall 回答「有没有漏掉真实颜色？」。低 Precision 的后果（大量错误预测）比低 Recall（漏报）更直接影响业务。因此本报告以 **Precision** 为主要准入判据。

## 4. 主要结论

### 4.1 尺寸分布的极端性

- 中位数面积仅 66 kpx（约 180×360），**40.8% 的图片面积 < 50 kpx**
- **64% 的图片短边 < 256px**，送入模型时需要上采样（信息有损）
- 44,077 种独特 (W,H) 组合，尺寸极度离散
- 中位数宽高比 0.438，典型人体 crop 为竖长形（H:W ≈ 2.3:1）

### 4.2 尺寸与性能的强正相关

- mIoU：极小图 (12.8kpx) 0.306 → 大图 (881kpx) 0.458，提升 **+48%**
- 颜色 Precision：极小图 0.733 → 大图 0.947，提升 **+29%**
- 颜色对尺寸的敏感度约为分割的 1.13 倍

### 4.3 推荐部署阈值

| 方案 | 条件 | 过滤比例 | 保留图片颜色 Prec |
|------|------|---------|-----------------|
| 保守 | short ≥ 80px 且 area ≥ 15kpx | ~5.6% | ~0.84 |
| **平衡（推荐）** | **short ≥ 96px 且 area ≥ 25kpx** | **~9.9%** | **~0.83** |
| 激进 | short ≥ 128px 且 area ≥ 40kpx | ~22.0% | ~0.81 |

规则：当 crop 图同时不满足短边和面积阈值时，跳过颜色属性预测，仅输出分割 mask。

### 4.4 混淆因子

数据源域差异不可忽略：ulpe 源（大图+高质量）mIoU 0.44，lip 源（最小图但标注一致）mIoU 0.36，ipc_seg 源（较大图但场景多样）mIoU 0.34。说明标注质量和域一致性至少与尺寸同等重要。

## 5. 文件说明

| 文件 | 说明 |
|------|------|
| `size_performance_analysis.html` | 完整统计报告（自包含，可直接浏览器打开） |
| `README.md` | 本说明文档 |

## 6. 复现命令

```bash
# 统计图片尺寸分布
python3 -c "
from PIL import Image
import os
# ... 遍历 images/ 目录读取所有图片尺寸
"

# 推理与指标计算
# 参见 PaddleSeg 实验配置：segment-color/experiment/PaddleSeg/experiments/
#   pp_liteseg_stdc2_clothes_color_2head_poseprior_6ch_256_optC_scratch/
```
