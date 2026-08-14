# Multi-Region Description 数据集总结

更新日期：2026-08-13

本文档是 `multi_region_description/data/` 中唯一应进入 Git 的文件。目录内的
JSONL、metadata 和分组清单均为本地生成数据，不随代码仓库发布。模型权重、训练日志、
推理输出和评估结果也不进入 Git。

## 1. 总览

当前数据目录包含 37 个文件，实际占用约 104 MiB。其中包括 16 组数据目录或数据视图，
以及 `data/` 根目录的一组早期 train/test 文件。数据本体全部是 JSON/JSONL 文本，未包含
原始图片；所有 JSONL 中的 `image` 都是 `/data/...` 形式的本机绝对路径。

因此，这些文件即使单独复制到其他机器，也不能直接用于训练。使用者还必须获得对应图片，
并重写 `image`、部分 `source_crop_path` 等路径。出于容量、数据授权、隐私、可移植性和避免
派生数据重复的考虑，仓库只保留生成与处理代码以及本报告。

全量检查结果：

- 所有 train/test JSONL 均可逐行解析，没有发现损坏的 JSON 行。
- 所有样本都包含训练所需的 `image`、`prompt`、`label` 字段。
- 所有样本的 `image` 字段均为本机绝对路径。
- 数据目录不包含图片本体，不能作为独立可下载的数据集使用。
- 多个目录只是同一批样本的标签格式、清洗方式或区域排序变体，不应简单相加理解为独立数据量。

## 2. 文件与样本统计

大小为 train/test 主文件合计，不含少量 metadata 或 split manifest；MiB 按 1024^2 字节计算。

| 数据集/视图 | 大小 | Train | Test | 合计 | 图像/帧数 | 区域数分布 |
|---|---:|---:|---:|---:|---:|---|
| `data/` 根目录（早期 Plan A） | 1.35 MiB | 2,983 | 331 | 3,314 | 3,314 | N1 1,489; N2 710; N3 496; N4 288; N5 331 |
| `ugipc-...-no-name` | 1.50 MiB | 2,983 | 331 | 3,314 | 3,314 | N1 1,270; N2 568; N3 383; N4 328; N5 765 |
| `ugipc-...-no-name-plan-b` | 1.77 MiB | 2,983 | 331 | 3,314 | 3,314 | 同上 |
| `ugipc-...-no-name-plan-b-sep` | 1.80 MiB | 2,983 | 331 | 3,314 | 3,314 | 同上 |
| `qwen-person-2to5-description-sep` | 4.20 MiB | 7,492 | 832 | 8,324 | 8,324 | N2 4,219; N3 2,159; N4 1,218; N5 728 |
| `qwen-person-2to5-description-loc` | 4.92 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-person-2to5-description-loc-sep` | 4.99 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-...-description-loc-order-random` | 4.92 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-...-description-loc-order-area-desc` | 4.92 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-...-description-loc-order-center-left-to-right` | 4.92 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-...-description-loc-order-center-top-to-bottom` | 4.92 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-person-2to5-cleaned-description-loc` | 6.13 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-...-cleaned-...-order-area-desc` | 6.13 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-...-cleaned-...-order-center-left-to-right` | 6.13 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-...-cleaned-...-order-center-top-to-bottom` | 6.13 MiB | 7,492 | 832 | 8,324 | 8,324 | 同上 |
| `qwen-person-2to6-cleaned-single-region` | 20.86 MiB | 23,185 | 2,576 | 25,761 | 8,713 帧 | 每条固定 N1 |
| `qwen-person-2to6-cleaned-dhash10-single-region` | 16.24 MiB | 16,547 | 1,839 | 18,386 | 6,277 帧 | 每条固定 N1 |

说明：

- `data/` 根目录与 `ugipc-...-no-name` 都有 3,314 条，但区域数分布不同，说明它们是不同阶段产物，不能假定内容完全相同。
- Qwen 2-to-5 的 11 个目录共享 8,324 个样本/图像和相同区域数分布，主要差异是描述来源是否清洗、label 文法以及 crop 排序。
- 单区域数据按 person region 展开，因此样本数大于源帧数。
- `qwen-person-2to6-cleaned-dhash10-single-region` 另有约 1.20 MiB 的 `split_groups.jsonl` 和 metadata，主文件表格没有计入这部分。

## 3. 数据来源与血缘

### 3.1 UGIPC / Florence 描述数据

源目录在代码中配置为：

```text
/data/work/MichaelYu/florence-data/ugipc-person-region-descriptions-yolo26m-no-name
```

`scripts/prepare_data.py` 从逐图 JSON 中读取图片路径、人物 bbox 和区域描述，执行以下处理：

1. 仅保留成功、描述非空且有四个量化 bbox 坐标的 region。
2. 对旧格式数据按完全相同的描述去重，重复描述只保留 crop 面积最大的 region。
3. region 超过 5 个时按面积保留最大的 5 个。
4. 没有有效 region 的图片跳过；单 region 样本允许保留。
5. 按固定随机种子切分 train/test，当前发布视图为 2,983/331。

`ugipc-...-no-name`、`plan-b` 和 `plan-b-sep` 共享数据来源，主要区别是 label 文法。

### 3.2 Qwen 多区域描述数据

Qwen 数据来自项目外的 `florence-data` 目录。2-to-5 数据只保留包含 2 至 5 个有效人物区域的图片，
使用 Qwen 成功生成的 description，并按 seed 42、test ratio 0.10 生成 7,492/832 切分。

基础视图包括：

- `description-sep`：隐式按输入顺序对应描述。
- `description-loc`：每段描述后回显四个 loc token，显式绑定 bbox。
- `description-loc-sep`：描述和 loc 后额外使用 `<sep>` 分隔。

排序视图由 `scripts/reorder_qwen_description_loc.py` 从 `description-loc` 联动重排 prompt
和 label 得到，包括随机顺序、面积降序、中心点从左到右、中心点从上到下。cleaned 目录使用清洗后的
Qwen description，但样本划分和区域数分布保持一致。

### 3.3 Qwen 单区域描述数据

`qwen-person-2to6-cleaned-single-region` 把每个源帧的 2 至 6 个人物 crop 展开为单独的
`REGION_TO_DESCRIPTION` 样本。metadata 记录：

- 总计 8,713 帧、25,761 个 region 样本。
- train/test 按源帧切分，避免同一帧跨 split。
- seed 为 42，请求测试比例为 0.10，实际比例约为 0.099996。
- 源帧 crop 数分布：2 个 4,219 帧；3 个 2,159 帧；4 个 1,218 帧；5 个 728 帧；6 个 389 帧。

`qwen-person-2to6-cleaned-dhash10-single-region` 是进一步去重并按相似帧组切分的版本：

- 总计 6,277 帧、18,386 个 region 样本。
- 5,760 个 split group，其中 292 个相似帧 cluster、5,468 个 singleton group。
- train/test 为 5,652/625 帧、16,547/1,839 样本。
- 最大 group 包含 18 帧或 37 个样本。
- metadata 报告跨 train/test 的相似帧 pair 为 0。
- 相似性使用文本、bbox IoU、整图 dHash 等阈值联合判断；完整阈值由生成脚本和本地 metadata 记录。

## 4. JSONL 格式

### 4.1 多区域基础 schema

```json
{
  "image": "/absolute/path/to/image.jpg",
  "prompt": "<REGIONS_TO_DESCRIPTIONS><loc_x1><loc_y1><loc_x2><loc_y2><sep>...",
  "label": "description 1<sep>description 2"
}
```

所有多区域视图至少包含 `image`、`prompt`、`label`。bbox 坐标量化到 0..999，并使用四个
`<loc_N>` token 表示。prompt 中的多个 bbox 使用 `<sep>` 分隔。

label 有三种文法：

| 名称 | 文法 | 对应目录 |
|---|---|---|
| `description-sep` | `desc1<sep>desc2...` | Plan A、Qwen `description-sep` |
| `description-loc` | `desc1<loc...>desc2<loc...>` | Plan B、Qwen `description-loc` 及排序变体 |
| `description-loc-sep` | `desc1<loc...><sep>desc2<loc...>` | `plan-b-sep`、Qwen `description-loc-sep` |

单 region 样本没有实际分隔符，所以其 label 可能只表现为 plain description 或
description+loc；这不是 schema 错误。

### 4.2 单区域扩展 schema

单区域文件除三个必需字段外，还保存可追踪信息：

```text
sample_id, frame_id, crop_key, crop_index,
bbox_loc_0_999, bbox_xyxy, source_json,
source_crop_path, description_model
```

dHash 分组版本额外包含 `similarity_group_id`。这些字段可用于按帧去重、错误分析和重新切分，
但其中路径仍依赖原始机器布局。

## 5. 可复现入口

仓库保留以下代码，不保留其生成物：

- `scripts/prepare_data.py`：生成多区域数据和三种 label 格式。
- `scripts/prepare_qwen_2to5_cleaned_loc.sh`：生成 cleaned Qwen 2-to-5 loc 数据。
- `scripts/reorder_qwen_description_loc.py`：生成区域排序变体。
- `scripts/prepare_single_region_data.py`：按源帧切分并生成单区域数据。
- `scripts/prepare_single_region_grouped_split.py`：按相似帧连通组生成防泄漏切分。

复现仍要求项目外的原始图片、逐 crop 描述 JSON、相似度分析结果以及相同的数据访问权限。
脚本中的默认路径是本机路径，跨机器运行时应通过命令行参数或环境变量覆盖。

## 6. Checkpoint 与其他生成物

虽然 checkpoint 不属于数据集目录，但它是本仓库最大的不可上传内容：

- `checkpoints/` 约 42 GiB，共 624 个文件。
- 其中 48 个 `model.safetensors`，单个约 884 MiB，合计约 41.4 GiB。
- 目录包含 16 组实验以及 epoch/final 快照，存在大量完整权重副本。
- 所有 checkpoint 均被 Git 忽略；需要通过模型仓库或对象存储单独发布。

训练日志、推理输出、评估 JSONL/图片、Python cache、demo PID/log 同样是可再生成或本地运行产物，
均不进入 Git。数据准备、训练、评估和 demo 源码继续进入 Git。

## 7. 发布前检查清单

若未来决定单独发布真实数据，应先完成：

1. 确认原始图片、人物检测结果以及 Florence/Qwen 派生描述的再分发授权。
2. 检查图片和描述中的人脸、身份、场所、设备屏幕及其他隐私信息。
3. 把绝对路径转换为数据包内相对路径，并验证每个引用文件存在。
4. 明确数据 license、用途限制、来源、生成模型版本及清洗规则。
5. 用稳定 ID 和校验和记录版本，避免多个排序/标签变体被误算成独立样本。
6. 保持按帧或相似帧 group 切分，防止近重复图像跨 train/test 泄漏。
7. 在独立数据托管服务发布，不把数据和 checkpoint 塞入普通 Git 历史。

## 8. Git 纳入策略

`multi_region_description/.gitignore` 执行以下策略：

- 忽略整个 `checkpoints/`。
- 忽略 `data/` 下所有内容，但显式保留本文件 `data/DATASET_SUMMARY.md`。
- 忽略 `logs/`、`outputs/`、`eval/logs/` 和 `eval/results/`。
- 忽略 badcase 的生成 JSON、Python cache、日志和 PID 文件。
- 保留 Python、Shell、HTML、CSS、JavaScript、测试、实验说明和本数据总结。

该策略保证代码仓库克隆体保持轻量，同时保留数据结构、规模、来源、处理方法和复现边界。
