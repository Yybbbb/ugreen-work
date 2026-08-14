# segment-color — 人体属性分割 + 颜色分类

基于 PaddleSeg (PP-LiteSeg + STDC2) 的人体属性语义分割与服装颜色分类项目。对人物裁剪图同时预测 **8 类分割 mask**（上下装、帽子、眼镜等）和 **上衣/下装颜色分类**（11 种标准色）。

---

## 项目概览

| 维度 | 说明 |
|------|------|
| **任务** | 人体属性语义分割（8类）+ 上衣/下装颜色分类（11类） |
| **模型** | PP-LiteSeg + STDC2 骨干，双头架构 |
| **框架** | [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) |
| **训练数据** | 71,345 训练样本 / 7,927 验证样本 |
| **输入** | RGB（3ch）或 RGB+Pose Heatmap（6ch），256×256 |
| **部署策略** | 小图拒绝 + 杂色检测置信度惩罚 |
| **关键指标** | mIoU=0.636, 上衣颜色 Acc=0.834, 下衣颜色 Acc=0.816 |

---

## 目录结构

```
segment-color/
├── README.md                         # 本文件
├── .gitignore                        # 忽略 data/ (68GB)、PaddleSeg clone、标注产物
├── yolo11n-pose.pt                   # YOLO11n-pose 人体检测模型权重
├── color_label_distribution_train.md # 训练集颜色标签分布统计文档
│
├── data/                             # 数据集（68GB，gitignore）
├── experiment/                       # 模型训练实验（PaddleSeg 框架）
├── extradata/                        # 外部原始数据（视频帧、检测图像）
├── labeling/                         # 标注工具链（SAM3、Qwen、人工审核）
├── logs/                             # 运行日志
├── refuse-answer/                    # 拒绝策略分析（小图/杂色）
├── scripts/                          # 流水线脚本（人物裁剪、标注调度）
└── statistics/                       # 推理分析（检出率 vs 像素占比）
```

---

## 各子目录详解

### 1. `data/` — 数据集

主数据集存储在 `LIP_clothes_accessory_unified/`，融合三个来源：

- **lip** — LIP（Look Into Person）公开数据集
- **ipc_seg_annotation** — IPC 安防场景标注
- **upper_lower_person_export** — 额外导出的人物裁剪

每张图片对应：
- `images/` — 原始 JPG 图片
- `annotations/` — JSON 格式分割标注（8 类 mask）
- `color-label/` — JSON 格式颜色标签（上衣/下装各 11 类）

辅助数据：
- `UPAR_rare_color/` — UPAR 数据集稀有颜色子集（Market1501、PA100k、PETA）
- `splits/` — 训练/验证集划分文件（`train.txt` 71,345 行 / `val.txt` 7,927 行）
- `cache/` — 预处理缓存（RGB-only 模式下的标注缓存）
- `pose_yolo26n/` — YOLO 姿态关键点数据

### 2. `experiment/PaddleSeg/` — 模型训练

PaddleSeg 完整 clone + 自定义实验代码，位于 `experiments/` 子目录：

| 子实验 | 说明 |
|--------|------|
| `clothes_color_rgb_only/` | RGB-only 变体（3 通道输入），数据集类 `ClothesColorRGBDataset` |
| `clothes_color_poseprior_stdc2/` | **主模型**（6 通道输入 = RGB + Pose Heatmap），包含模型定义 `PPLiteSegWithColorHeads`、损失函数、训练/评估脚本 |
| `clothes_color_zeropose_ablation/` | 消融实验（Pose 输入置零），包含推理、评估、HTML 错误分析 |

**核心模型 `PPLiteSegWithColorHeads`**：
- **Backbone**：STDC2（高效语义分割网络）
- **Segmentation Head**：8 类输出（background、upper、lower、dress、hat、glasses、mask、hair）
- **Color Heads**：两个独立分类头，各输出 11 类颜色（black / white / gray / red / yellow / green / blue / purple / pink / orange / brown）+ unknown
- **输入**：256×256，RGB（3ch）或 RGB+Pose（6ch）
- **训练**：OptC 优化器，从头训练（scratch）

### 3. `extradata/` — 外部原始数据

- **视频帧提取**：`extract_frames_batch.py` 使用 ffmpeg 从 .dat 视频文件中每 2 秒抽 1 帧
- **物体检测场景帧**：`object_detection_0309-0429/` 包含 14 个场景目录（公园、欧洲城、五金店、展会、城中村、猫咖、商场、宜家、交通枢纽、住宅区等）
- **UPAR 数据集集成**：`upar-dataset/` 含原始标注、稀有颜色筛选脚本（28,376 样本）、杂色/稀有颜色验证样本
- **辅助工具**：`delete_small_images.py`（移除短边 < 48px 的小图）

### 4. `labeling/` — 标注工具链

**SAM3 标注（主方案）**：
- `annotate_ipc.py` — IPC 训练集全量标注脚本，通过 HTTP 向 14 个 SAM3 服务实例（7 GPU，端口 8020–8033）发送约 30 个 prompt/张的标注请求，支持断点续传、超时重试、置信度阈值
- `server.py` — HTTP 标注审核服务器

**Qwen VL 标注（辅助方案）**：
- `qwen_upper_lower_color.py` — 基于 Qwen 视觉语言模型的颜色直接标注
- `qwen_upper_lower_color_prompt.txt` — Qwen 系统提示词

**数据质量**：
- `analyze_label_distribution.py` — 颜色标签分布统计
- `generate_annotation_review.py` / `annotation_review.html` — 人工审核页面
- `server.log` — 审核服务器日志

### 5. `scripts/` — 流水线脚本

端到端数据处理流水线的核心脚本：

- `crop_persons_from_frames.py` — YOLO11n-pose 人体检测 → 人物裁剪 + 姿态标注
- `person_crop_pipeline.py` — 通用版 YOLO 姿态人体裁剪流水线（可配置参数）
- `sam3_annotate_upar_rare_color.py` — 调用 SAM3 批量 API 标注 UPAR 稀有颜色样本

**典型流水线**：原始视频 → ffmpeg 抽帧 → YOLO 人体检测与裁剪 → SAM3 属性标注 → 数据集构建 → 模型训练 → 推理分析

### 6. `statistics/` — 推理分析

分析模型在每个属性上的"检出"表现（是否存在，而非像素精度）：

- `predict_detection.py` — 对验证集 7,927 张图推理，输出逐图检出 CSV
- `gen_detection_html.py` — 读取 CSV，生成自包含 HTML 分析报告
- `detection_analysis_report.html` — 最终报告（含汇总表、像素占比 vs Precision 图表、Top-20 FP/FN Badcase 展示）
- `detection_analysis_report_train.html` — 训练集分析报告
- `per_image_detection.csv` / `per_image_detection_train.csv` — 逐图检出数据
- `badcase/` — Badcase 图片的 GT/Pred mask（NPY + PNG）

**核心分析维度**：属性像素占比（GT/Pred ratio）与检出 Precision/Recall 的关系，为部署阶段设置置信度阈值提供依据。

### 7. `refuse-answer/` — 拒绝策略

部署阶段的两种质量保证策略：

#### `smallcrop/` — 小图拒绝
分析人物裁剪图尺寸与颜色预测质量的关系，给出数据驱动的尺寸准入阈值：
- **推荐阈值**：短边 < 96px **且** 面积 < 25 kpx 时拒绝预测
- **结论**：基于 53,269 张 train+val 图片统计分析，拒绝 40.8% 的小图可将整体颜色 Precision 提升约 3 个百分点
- 输出：`size_performance_analysis.html`（自包含报告）

#### `multicolor/` — 杂色检测
检测条纹、方格、拼色等非纯色服装，对颜色分类结果施加置信度惩罚：
- **方法**：Lab 颜色空间直方图峰值检测 + RLE 游程编码纹理分析
- **速度**：约 5ms/样本
- **核心模块**：
  - `multicolor_detector.py` — 检测器对外接口
  - `histogram_peak.py` — 二维直方图构建与峰值检测
  - `texture_rle.py` — RLE 纹理分析（条纹/方格/拼色判定）
  - `tune_params.py` / `tune_results.json` — 参数网格搜索
  - `final_eval.py` / `final_eval_results.json` — 最终评估结果
  - `experiment_report.html` — 实验报告

### 8. `logs/` — 运行日志

- `sam3_upar_rare_color_20260623_175839.log` — SAM3 标注运行日志（3.5MB）

---

## 颜色类别

11 种标准分类颜色：

| black | white | gray | red | yellow | green | blue | purple | pink | orange | brown |
|-------|-------|------|-----|--------|-------|------|--------|------|--------|-------|

外加 `unknown`（无法确定）。训练集颜色分布详见 [color_label_distribution_train.md](color_label_distribution_train.md)。

### 关键分布（训练集 71,345 样本）

| 上衣 | 占比 | 下装 | 占比 |
|------|------|------|------|
| black | 19.89% | black | 41.10% |
| white | 18.86% | blue | 20.89% |
| gray | 11.42% | gray | 10.49% |
| red | 9.22% | brown | 8.84% |

---

## 端到端流水线

```
[原始 .dat 视频]
    │  extradata/extract_frames_batch.py (ffmpeg, 1帧/2秒)
    ▼
[视频帧]
    │  scripts/crop_persons_from_frames.py (YOLO11n-pose)
    ▼
[人物裁剪图]
    │  labeling/annotate_ipc.py (SAM3, 14实例并行)
    │  或 labeling/qwen_upper_lower_color.py (Qwen VL)
    ▼
[标注数据集] → data/LIP_clothes_accessory_unified/
    │
    ▼
[模型训练] → experiment/PaddleSeg/
    │  PP-LiteSeg + STDC2, 双头（分割+颜色）
    ▼
[模型检查点]
    │  statistics/predict_detection.py → 检出分析
    │  refuse-answer/ → 拒绝策略
    ▼
[部署]  小图拒绝阈值 + 杂色置信度惩罚
```

---

## 关键文件说明

| 文件 | 用途 |
|------|------|
| [yolo11n-pose.pt](yolo11n-pose.pt) | YOLO11n-pose 模型权重，用于人物检测与姿态估计 |
| [color_label_distribution_train.md](color_label_distribution_train.md) | 训练集 12 类颜色标签分布详细统计 |
| [statistics/detection_analysis_report_train.html](statistics/detection_analysis_report_train.html) | 训练集检出分析报告（自包含 HTML） |
| [statistics/IMPLEMENTATION_PLAN.md](statistics/IMPLEMENTATION_PLAN.md) | 检出分析模块实现计划 |
| [refuse-answer/smallcrop/README.md](refuse-answer/smallcrop/README.md) | 小图拒绝策略分析报告 |
| [refuse-answer/multicolor/DESIGN.md](refuse-answer/multicolor/DESIGN.md) | 杂色检测模块设计文档 |

---

## 环境要求

- **深度学习框架**：PaddlePaddle（PaddleSeg）
- **人物检测**：YOLO11n-pose（Ultralytics）
- **标注**：SAM3（Segment Anything 3）服务、Qwen VL 模型
- **数据分析**：NumPy、Pandas、Matplotlib、Chart.js
- **视频处理**：ffmpeg（帧提取）
