# segment-color 数据集与产物总结报告

> 本文件是 segment-color 仓库中**唯一入库的数据说明**。仓库内所有数据集、模型权重、标注产物均已在 [.gitignore](.gitignore) 中排除,需通过共享存储 / rsync 单独同步。本报告记录它们的规模、结构、来源与恢复方式,使得在一台干净的机器上 clone 代码后,能够据此重建完整的数据环境。
>
> - 统计时间:2026-08-13
> - 统计口径:`du -sh`(磁盘占用)+ `find -type f | wc -l`(文件数)
> - 仓库总占用:**95 GB**,其中入库代码约 **43 MB / 144 文件**

---

## 1. 总览

| 目录 | 占用 | 文件数 | 是否入库 | 性质 |
|------|-----:|-------:|:--------:|------|
| [data/](data/) | 78 GB | 596,188 | 否 | 主数据集 + 训练缓存 |
| [experiment/PaddleSeg/](experiment/PaddleSeg/) | 15 GB | 29,706 | 否 | 第三方框架 + 训练权重 |
| [extradata/](extradata/) | 1.2 GB | 187,280 | 部分 | UPAR 外部数据集(仅脚本入库) |
| [labeling/](labeling/) | 627 MB | 53,333 | 部分 | 标注产物(仅工具代码入库) |
| [refuse-answer/](refuse-answer/) | 76 MB | 705 | 部分 | 拒答策略实验(代码 + HTML 报告入库) |
| [statistics/](statistics/) | 42 MB | 133 | 部分 | 推理统计(代码 + HTML 报告入库) |
| `yolo11n-pose.pt` | 6.0 MB | 1 | 否 | YOLO11n-pose 官方权重 |
| [logs/](logs/) | 3.5 MB | 1 | 否 | 运行日志 |
| [scripts/](scripts/) | 88 KB | — | 是 | 流水线脚本 |

数据占比 99.95%,`data/` 与 `experiment/PaddleSeg/` 两项即占 93 GB。

---

## 2. 主数据集 `data/LIP_clothes_accessory_unified` — 69 GB

项目的核心数据集,由三个来源融合而成。元信息见该目录下 `summary.json`。

- **原始构建路径**:`/data0/work/LuoWeiXing/LIP_clothes_accessory_unified`
- **schema_version**:1.0
- **图像存储方式**:`hardlink`(硬链接引用源数据,迁移时需注意跨设备失效)
- **总样本数**:89,221
- **分割类别(7 类 + 背景 = 8 类)**:`upper` `lower` `dress` `hat` `glasses` `mask` `hair`

### 2.1 数据来源构成

| 来源 | 样本数 | 说明 |
|------|-------:|------|
| `lip` | 40,462 | LIP (Look Into Person) 公开数据集 |
| `ipc_seg_annotation` | 34,010 | IPC 安防场景人工标注 |
| `upper_lower_person_export` | 14,749 | 额外导出的人物裁剪图 |

### 2.2 目录构成

| 子目录 | 占用 | 文件数 | 说明 |
|--------|-----:|-------:|------|
| `cache/` | 64 GB | — | 训练预处理缓存,**可完全重新生成** |
| `images/` | 2.9 GB | 89,221 | 人物裁剪图,按三个来源分子目录 |
| `annotations/` | 950 MB | 89,221 | 分割标注 |
| `color-label/` | 620 MB | 53,269 | 上下装颜色标签 |
| `pose_yolo26n/` | 343 MB | 108 | YOLO26n 姿态预标注 |
| `lists/` | 20 MB | 4 | `all.txt` + 三个来源各一份清单 |
| `splits/` | 8.6 MB | — | 训练/验证划分 |
| `sam3_unified_completion.jsonl` | 9.3 MB | 1 | SAM3 补标日志 |
| `sam3_hair_completion.jsonl` | 9.2 MB | 1 | SAM3 头发补标日志 |
| `all-data-clean-image.txt` | 8.7 MB | 1 | 清洗后图像清单 |

`cache/` 占了主数据集的 93%,细分为:

| 缓存 | 占用 | 内容 |
|------|-----:|------|
| `clothes_accessory_poseprior/pose` | 48 GB | 姿态 heatmap(6 通道输入用) |
| `clothes_accessory_poseprior/labels_8cls_hair` | 8.0 GB | 8 类含头发标签 |
| `clothes_accessory_poseprior/labels` | 8.0 GB | 基础标签 |

**这 64 GB 无需同步**,由 `images/` + `annotations/` + `pose_yolo26n/` 经预处理脚本重新生成即可。

### 2.3 `pose_yolo26n/` 姿态预标注

| 文件 | 占用 | 说明 |
|------|-----:|------|
| `annotations.jsonl` | 225 MB | 逐样本关键点 |
| `annotations.csv` | 99 MB | 同上,CSV 格式 |
| `lists/` | 16 MB | 按质量分级的清单 |
| `visual_samples/` | 3.8 MB | 可视化抽样 |
| `summary.json` | 4 KB | 统计摘要 |

### 2.4 划分定义 `splits/clothes_accessory_poseprior`

由 `summary.json` 记录的划分参数:

- `val_ratio`: 0.12
- `seed`: 20260518
- `kept_qualities`: `database`, `reid`(过滤掉其他质量等级)

| 划分 | 总数 | reid | database |
|------|-----:|-----:|---------:|
| all | 53,269 | 43,405 | 9,864 |
| train | 46,878 | 38,197 | 8,681 |
| val | 6,391 | 5,208 | 1,183 |

按来源 × 质量的细分(all):

| 来源 | database | reid |
|------|---------:|-----:|
| `lip` | 4,093 | 17,135 |
| `ipc_seg_annotation` | 3,860 | 16,054 |
| `upper_lower_person_export` | 1,911 | 10,216 |

### 2.5 分割掩码覆盖情况

各来源中带有效 mask 的样本数(摘自 `summary.json` 的 `attribute_has_mask_counts`):

| 属性 | lip | ipc_seg_annotation | upper_lower_person_export |
|------|----:|-------------------:|--------------------------:|
| upper | 36,654 | 29,765 | 14,749 |
| lower | 25,402 | 22,590 | 14,749 |
| hair | 28,072 | 24,531 | 12,873 |
| hat | 10,421 | 14,294 | 1,600 |
| glasses | 2,347 | 15,425 | 4,676 |
| dress | 1,642 | 15,841 | — |
| mask | 626 | 1,363 | 1,060 |

`mask`(口罩)是显著的长尾类别,三个来源合计仅 3,049 个正样本。

### 2.6 SAM3 补标记录

`sam3_unified_completion_summary.json` 记录了头发类别的补标过程:

- 完成时间:2026-05-20
- 类别:`hair`
- 服务:8 个 SAM3 实例(`127.0.0.1:9010-9017`)
- 结果:48,759 请求全部成功

---

## 3. 其他数据子集

### 3.1 `data/cache/rgb_only` — 8.6 GB

RGB 三通道输入的训练缓存,与主数据集的 `pose` 缓存并列。同样**可重新生成**。

### 3.2 `data/object_detection_0309-0429` — 446 MB

2026-03-09 至 04-29 的检测场景数据。

| 子目录 | 占用 |
|--------|-----:|
| `images/` | 277 MB |
| `annotations/` | 95 MB |
| `color-label/` | 58 MB |
| `subject-label/` | 11 MB |
| `sam3_..._completion.jsonl` | 4.6 MB |

附带 `subject_label_random50_report.html`(随机 50 例人工核查报告)与 `qwen_subject_carry_annotate.py`(Qwen 携带物标注脚本,已入库)。

### 3.3 `data/UPAR_rare_color` — 438 MB

针对稀有颜色补充的样本,用于缓解颜色类别长尾。

| 子目录 | 占用 |
|--------|-----:|
| `images/` | 178 MB |
| `annotations/` | 152 MB |
| `color-label/` | 104 MB |
| `rare_color_single_color.txt` | 4.9 MB |

### 3.4 划分清单 `data/splits` 与 `data/splits-new`

| 文件 | 行数 |
|------|-----:|
| `splits/train.txt` | 71,345 |
| `splits/val.txt` | 7,927 |
| `splits-new/train.txt` | 84,295 |
| `splits-new/val.txt` | 9,366 |

`splits/` 对应 README 中记录的 71,345 / 7,927 基线;`splits-new/` 是并入 UPAR 稀有色与检测场景数据后的扩充版本。

---

## 4. `experiment/PaddleSeg` — 15 GB

第三方框架的本地 clone,**不应入库**,在目标机器上重新获取:

```bash
cd experiment && git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

其中训练产物需单独从对象存储同步:

| 子目录 | 占用 | 内容 |
|--------|-----:|------|
| `experiments/` | 7.1 GB | 各组实验的 checkpoint |
| `output/` | 7.0 GB | 训练输出与最优权重 |
| `onnx/` | 282 MB | 导出的 ONNX 模型 |
| 框架自身代码 | ~70 MB | docs / contrib / paddleseg 等 |

模型配置:PP-LiteSeg + STDC2 骨干,双头架构(分割 + 颜色分类),输入 256×256,支持 RGB(3ch)与 RGB+Pose Heatmap(6ch)。

**基线指标**:mIoU = 0.636,上衣颜色 Acc = 0.834,下装颜色 Acc = 0.816。

---

## 5. 标注与实验产物

### 5.1 `labeling/` — 627 MB / 53,333 文件

标注工具链(SAM3 + Qwen + 人工审核)的产物。文件数过多,不适合 git,用 rsync 同步。

| 子目录 | 占用 | 文件数 | 入库 |
|--------|-----:|-------:|:----:|
| `annotation/` | 620 MB | 53,270 | 否 |
| `review_results/` | 1.4 MB | — | 否(其中 `ANALYSIS_REPORT.md` 已排除) |
| `images/` | 672 KB | — | 否 |
| `org_annotation/` | — | — | 否 |
| 工具代码(`server.py` 等 9 个 .py + tests) | ~200 KB | — | **是** |
| `annotation_review.html` | 4.6 MB | 1 | 是 |

### 5.2 `extradata/` — 1.2 GB / 187,280 文件

UPAR 外部行人属性数据集,包含三个公开子集:

| 子集 | 说明 |
|------|------|
| `Market1501` | 行人重识别数据集 |
| `PA100k` | 行人属性数据集 |
| `PETA` | 行人属性数据集 |
| `annotations` | 统一标注 |

这些是**公开数据集**,应从官方渠道获取而非仓库分发。抽取与清洗脚本已入库:

- `extradata/scripts/` — 抽帧、小图过滤、人物核查、Qwen 颜色标注与清洗、短边分布统计(6 个脚本)
- `extradata/upar-dataset/scripts/extract_rare_colors.py` — 稀有色抽取

清洗结果 `qwen_color_clean_results.json`(3.8 MB)、`person_check_results.json`(1.2 MB)、`color_check.html`(5.2 MB)、`数据明细.xlsx`(212 KB)均可由脚本重新生成,已排除。

### 5.3 `refuse-answer/` — 76 MB / 705 文件

部署侧拒答策略研究,分两条线:

| 子目录 | 占用 | 内容 |
|--------|-----:|------|
| `smallcrop/` | 44 MB | 小图拒答:`infer_train_rgb/` 推理结果 + 尺寸-性能分析 |
| `multicolor/` | 32 MB | 杂色检测:v1→v4 四轮迭代 |

文件类型:569 jpg(样本图,已排除)、51 py(代码,入库)、19 json、7 html(报告,入库)、5 md。

v1→v4 的设计文档 `DESIGN.md` / `RULES.md` / `IMPLEMENTATION_PLAN.md` 与各轮 `eval_report.html` 均已入库,构成完整的迭代记录。

### 5.4 `statistics/` — 42 MB / 133 文件

检出率 vs 像素占比的推理分析。

| 文件 | 占用 | 入库 |
|------|-----:|:----:|
| `per_image_detection_train.csv` | 22 MB | 否 |
| `per_image_detection.csv` | 2.9 MB | 否 |
| `badcase/`(80 npy + 40 png) | 9.5 MB | 否 |
| `detection_analysis_report.html` | 6.5 MB | **是** |
| `detection_analysis_report_train.html` | 1.1 MB | **是** |
| `gen_detection_html.py` / `predict_detection.py` | 84 KB | **是** |
| `IMPLEMENTATION_PLAN.md` | 12 KB | 是 |

两个 CSV 是逐图推理的原始记录,由 `predict_detection.py` 重新生成;HTML 报告已包含全部结论,故报告入库、原始数据排除。

---

## 6. 颜色标签分布

完整统计见 [color_label_distribution_train.md](color_label_distribution_train.md)(已入库)。要点:

- 统计日期 2026-06-25,基于 `data/splits/train.txt` 的 71,345 样本,解析失败 0
- 颜色体系:12 个标准类(`unknown` `black` `white` `gray` `red` `yellow` `green` `blue` `purple` `pink` `orange` `brown`),已过滤 `beige` `maroon` `silver` `gold` `tan` `khaki` 等非标准标签

| 维度 | 上衣 | 下装 |
|------|-----:|-----:|
| 有效标注 | 71,215 | 71,285 |
| 缺失 | 130 | 60 |
| 多色样本(>1 色) | 7,309 (10.26%) | 1,414 (1.98%) |

主色 Top3 —— 上衣:black 19.89% / white 18.86% / gray 11.42%;下装:black 41.10% / blue 20.89% / gray 10.49%。下装的 black 集中度显著更高,`orange` `purple` 在两个维度上都低于 1.1%,是主要长尾。

---

## 7. 在新机器上重建数据环境

```bash
# 1. clone 代码(约 43 MB)
git clone <repo> segment-color && cd segment-color

# 2. 获取 PaddleSeg 框架
cd experiment && git clone https://github.com/PaddlePaddle/PaddleSeg.git && cd ..

# 3. 下载 YOLO11n-pose 官方权重
#    https://github.com/ultralytics/assets/releases → yolo11n-pose.pt 放到仓库根目录

# 4. 从共享存储同步数据(按需,cache 可跳过)
rsync -av --progress <shared>/LIP_clothes_accessory_unified/{images,annotations,color-label,pose_yolo26n,lists,splits} \
    data/LIP_clothes_accessory_unified/
rsync -av --progress <shared>/{object_detection_0309-0429,UPAR_rare_color,splits,splits-new} data/

# 5. 重新生成预处理缓存(替代 64 GB + 8.6 GB 的 cache 同步)
#    使用 experiment/ 下的预处理脚本重建 cache/clothes_accessory_poseprior 与 cache/rgb_only

# 6. 训练权重按需从对象存储拉取
rsync -av <shared>/PaddleSeg_output/ experiment/PaddleSeg/output/
```

**同步量对比**:全量 95 GB → 必要数据约 **5.5 GB**(主数据集非缓存部分 4.9 GB + 两个补充子集 0.9 GB),缓存与权重按需重建/拉取。

### 迁移注意事项

1. **硬链接失效** —— 主数据集 `image_storage` 为 `hardlink`,原始构建路径在 `/data0/work/LuoWeiXing/`。跨设备 rsync 时硬链接会被展开为独立文件副本,`images/` 的实际传输量会大于 2.9 GB 的表观占用。
2. **绝对路径** —— `summary.json`、`splits/*/summary.json` 中记录的 `dataset_root`、`lists`、`train_path` 等均为 `/data0/work/LuoWeiXing/...` 绝对路径,在新环境需改写或通过软链接对齐。
3. **`cache/` 与源数据的一致性** —— 缓存基于特定版本的 `annotations/` 与 `pose_yolo26n/` 生成,若源数据更新需重建缓存,否则会静默使用过期标签。

---

## 8. 排除规则说明

[.gitignore](.gitignore) 的排除逻辑,以及为何这样切分:

| 规则 | 排除内容 | 理由 |
|------|---------|------|
| `data/` | 78 GB | 数据集,走共享存储 |
| `experiment/PaddleSeg/` | 15 GB | 第三方仓库 + 训练权重 |
| `*.pt` `*.pdparams` `*.onnx` | 模型权重 | 从官方渠道下载或对象存储拉取 |
| `labeling/{annotation,review_results,images,org_annotation}/` | 627 MB / 5.3 万文件 | 文件数过多,git 索引开销大 |
| `extradata/upar-dataset/data/` 等 | 1.2 GB / 18.7 万文件 | 公开数据集,官方获取 |
| `extradata/*.{html,json,xlsx}` | ~10 MB | 脚本可重新生成 |
| `refuse-answer/**/*.{jpg,png,jsonl}` | 样本图与推理结果 | 保留 .py / .md / .html,结论已在报告中 |
| `statistics/*.csv` `statistics/badcase/` | 32 MB | 保留 HTML 报告与生成脚本 |
| `!DATASET_REPORT.md` | — | 确保本报告入库 |

**保留在库内的"重产物"**:各类 HTML 实验报告合计约 20 MB(`detection_analysis_report.html` 6.5 MB、`annotation_review.html` 4.6 MB、四轮 `eval_report.html` 约 10 MB)。它们体积偏大但承载了实验结论,且无法从代码重新生成(依赖已排除的原始数据),因此选择入库。若后续要压缩仓库体积,这是首选目标。
