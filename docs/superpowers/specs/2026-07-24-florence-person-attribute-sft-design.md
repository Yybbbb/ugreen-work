# Florence2 人物属性 SFT 数据与训练设计

## 1. 目标与范围

本实现以 `/data1/work/MichaelYu/florence-attibute/pretrained/Florence-2-base` 为只读初始权重，使用 `/data1/work/MichaelYu/florence-attibute/data` 中已经去重、划分的数据，构建可复现的人物属性 SFT 数据和 4 卡全参数训练流程。

人物主任务使用 Florence2 原生 `<REGION_TO_CATEGORY>`：输入完整 frame 和一个人物区域，输出 Qwen 英文人物属性 caption。训练数据为 30,000 条 person train；dev 为 2,000 条独立 session 样本；test 为独立测试集 4,328 条，只在训练完成后使用。

为缓解原生区域描述格式遗忘，另从未进入 train/dev 的 reserve 中选择 1,000 条人物区域，使用冻结的 Florence2-base 和原生 `<REGION_TO_DESCRIPTION>` 离线生成 target。正式 SFT 混合 30,000 条人物数据和 1,000 条 replay，只训练 1 epoch。

本阶段不实现 RL、属性语义 embedding 评测或 COCO replay。训练过程输出逐样本 dev 预测，供后续属性评测模块使用。

## 2. 实现结构

新增脚本按职责拆分：

```text
florence-attibute/
  scripts/
    person_sft_data.py              # 共享数据 schema、bbox 量化和 JSONL 工具
    prepare_person_sft_data.py      # scene JSON + split manifest -> person JSONL
    generate_region_replay.py       # reserve 采样和 base 模型离线生成 replay
    train_person_attribute_sft.py   # torchrun/DDP 全参数训练、dev、checkpoint
    validate_sft_checkpoint.py      # 新进程重载并验证 processor/model
  data/
    prepared/
      train.jsonl
      dev.jsonl
      test.jsonl
      rl.jsonl
      native_replay_selection.jsonl
      native_replay.progress.jsonl
      native_replay.jsonl
      metadata.json
  tests/
    test_person_sft_data.py
    test_prepare_person_sft_data.py
    test_generate_region_replay.py
    test_train_person_attribute_sft.py
```

`person_sft_data.py` 不依赖 torch 或 transformers，使数据预处理和多数测试能在轻量 Python 环境运行。模型相关依赖只在 replay 生成和训练入口实际调用时导入。

## 3. Person 数据前处理

### 3.1 输入映射

每个 split 以 `data/manifests/<split>.jsonl` 为权威人物清单，以 `data/<split>/<scene>/*.json` 为 frame 标注来源。manifest 中的 `source_json_relative_path` 定位 frame JSON，稳定的 `crop_index` 定位人物。split JSON 已删除未入选 crop 并压缩 `crops` 列表，因此 `crop_position` 只用于保留原候选池 sample ID，不能作为物化 JSON 的列表下标。

预处理逐条验证：

- manifest 的 `sample_id`、scene、session 和 caption 非空；
- frame JSON 可读取，`image.path` 指向可读取的完整 frame；
- `crop_index` 在 frame JSON 中唯一匹配，crop caption 与 manifest caption 完全一致；
- crop annotation 状态成功，`expanded_bbox_xyxy` 为四个有限数且面积为正；
- 图像宽高为正，bbox 裁剪到图像边界后仍有正面积。

### 3.2 bbox 量化与 prompt

不复用 JSON 中已有的归一化 bbox。对裁剪到图像边界的 `expanded_bbox_xyxy=[x1,y1,x2,y2]`，按 Florence2 规则重新量化：

```text
qx = clamp(floor(x / (width / 1000)), 0, 999)
qy = clamp(floor(y / (height / 1000)), 0, 999)
```

人物 prompt 固定为：

```text
<REGION_TO_CATEGORY><loc_qx1><loc_qy1><loc_qx2><loc_qy2>
```

输出 JSONL 每行 schema：

```json
{
  "schema_version": "florence_person_sft_v1",
  "sample_id": "scene/frame.json#0",
  "split": "train",
  "task": "<REGION_TO_CATEGORY>",
  "image": "/absolute/path/to/frame.jpg",
  "prompt": "<REGION_TO_CATEGORY><loc_1><loc_2><loc_3><loc_4>",
  "label": "An adult ...",
  "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
  "bbox_loc_0_999": [1, 2, 3, 4],
  "scene": "scene_name",
  "session": "scene/session",
  "source_json": "scene/frame.json",
  "crop_position": 0,
  "crop_index": 1,
  "scale": "small",
  "attributes": {}
}
```

`train/dev/test/rl` 分别生成独立 JSONL。脚本不重新划分、不复制图像、不读取 reserve 作为人物监督。写入使用同目录临时文件，完成数量、唯一 ID 和 hash 校验后原子替换正式文件。

## 4. REGION_TO_DESCRIPTION Replay

### 4.1 确定性采样

replay 候选只来自 `data/manifests/reserve.jsonl`，不得与 train/dev/test 混用。当前 reserve 包含：

| 场景 | 人物 | frame |
|---|---:|---:|
| `company_surveillance_mp4_adaptive` | 610 | 610 |
| `tradeshow` | 20,992 | 5,989 |

随机种子固定为 `20260720`，每个场景选择 500 条，共 1,000 条。每个 frame 至多选择一个人物；排序同时考虑原有 `selection_score`、尺度覆盖、属性丰富度和确定性随机 tie-break。选择结果先固定为 `native_replay_selection.jsonl`，后续重跑不得重新抽样。

每条 replay 使用完整 frame 和该人物的 `expanded_bbox_xyxy`，prompt 为：

```text
<REGION_TO_DESCRIPTION><loc_x1><loc_y1><loc_x2><loc_y2>
```

### 4.2 Base 模型生成

生成器只读加载 `pretrained/Florence-2-base` 的 processor 和模型，使用本地动态代码、fast tokenizer、eager attention 和 BF16（支持 CUDA 时）。默认 greedy 参数：

```text
num_beams=1
do_sample=false
max_new_tokens=64
```

生成文本通过 `processor.post_process_generation(..., task="<REGION_TO_DESCRIPTION>")` 后处理。当前 checkpoint 返回 `{task: pure_text}`，共享解包函数同时兼容直接字符串返回；解包后得到 replay label。空输出、异常 loc token、重复 sample ID 或图像读取失败均记录明确错误并使最终汇总失败。

每完成一条就在 `native_replay.progress.jsonl` 写入 sample ID、prompt、原始生成文本、后处理 label 和生成配置。重启时校验已有行并跳过成功项；进度达到 1,000 条后按 selection manifest 的固定顺序原子生成 `native_replay.jsonl`。

## 5. SFT 训练

### 5.1 数据与分布式执行

训练入口由 `torchrun --nproc_per_node=4` 启动。训练集由 30,000 条 `train.jsonl` 和 1,000 条 `native_replay.jsonl` 组成，按固定种子做一次确定性混合；不复制样本。使用 `DistributedSampler` 保证各 rank 的样本分片一致且每 epoch 可复现。

dev 只读取 2,000 条 person 数据，不混 replay。test 不在训练循环内运行，也不参与 checkpoint 选择。

默认配置：

| 配置 | 数值 |
|---|---:|
| epoch | 1 |
| GPU | 4 |
| 每卡 batch | 4 |
| 梯度累积 | 4 |
| global batch | 64 |
| optimizer step | 约 485 |
| precision | BF16 AMP |
| optimizer | AdamW |
| weight decay | 0.01 |
| label smoothing | 0.05 |
| max grad norm | 1.0 |
| warmup | 总 optimizer step 的 5% |
| scheduler | cosine，最低为峰值学习率的 10% |
| prompt max length | 1024，包含 577 个图像 token 预算 |
| target max length | 64 |
| eval interval | 50 optimizer step |
| train num workers | 6 per rank |
| dev num workers | 0 per rank |
| prefetch factor | 1 batch per worker |

梯度累积期间仅在每个累积组的最后一个 micro-step 同步 DDP 梯度。完整组将 loss 除以 4；epoch 尾部不足 4 个 micro-step 时，将 loss 除以尾组实际 micro-batch 数后执行一次 optimizer step。scheduler 总步数使用向上取整结果。

### 5.2 全参数和分组学习率

所有参数保持 `requires_grad=True`，并按参数名互斥分组：

- `vision_tower.*`：`1e-7`；
- `image_projection`、视觉 projection norm、视觉 position/temporal embedding：`5e-7`；
- 其余 language encoder-decoder、embedding 和 LM head：`1e-6`。

启动时打印每组参数张量数和参数量，并断言每个可训练参数恰好属于一个组。bias 和 normalization 参数不使用 weight decay，其余参数使用 0.01。

### 5.3 Loss、dev 和日志

processor 批量处理完整 frame 与 raw prompt；label 使用 processor tokenizer，padding ID 替换为 `-100`。训练使用 logits 上的交叉熵并设置 `label_smoothing=0.05`。

每 50 optimizer step：

1. 在完整 dev 上分布式计算 teacher-forced loss；
2. rank 0 对固定、分层的 dev 子集执行 greedy generation；
3. 输出 `dev_predictions_step_<step>.jsonl`；
4. 汇总平均词数、18-24 词比例、P95 词数、单句率、人物主体开头率和空输出率。

top 3 checkpoint 暂按 dev loss 选择。属性准确率、开集相似度和背景分类需要独立评测模块，本阶段保留逐样本 caption、attributes 和元数据作为后续输入。

### 5.4 Checkpoint 与恢复

每个 checkpoint 保存：

- `model.save_pretrained()` 的完整全参数权重；
- processor/tokenizer 和 Florence 动态模型文件；
- optimizer、scheduler、AMP scaler；
- epoch、micro-step、optimizer step、best checkpoint 列表；
- Python、PyTorch CPU/CUDA、sampler 随机状态；
- 完整 CLI 配置和输入 JSONL SHA-256。

`--resume-from` 校验模型、数据 hash、world size、batch、梯度累积和 scheduler 总步数后恢复。关键配置不一致时拒绝继续，避免在不同数据或 global batch 上静默续训。

## 6. 并发、故障与文件安全

只有 rank 0 创建运行目录、写日志、生成 dev 预测和保存 checkpoint；其他 rank 在 barrier 后继续。任一 rank 抛出异常时都在 `finally` 中销毁 process group。

预处理、replay 最终汇总和训练元数据均采用临时文件加 `os.replace`。原始 `data/<split>`、manifest、reserve 和 `pretrained/Florence-2-base` 不被修改。训练输出统一写入新的 `artifacts/sft/<run_name>`。

提供以下低成本控制：

- `--max-train-samples` 和 `--max-dev-samples`：限制数据量；
- `--max-optimizer-steps`：训练 smoke test；
- `--num-workers 0`：调试图像错误；
- `--no-save`：smoke test 不保存大权重；
- `--validate-only`：只验证数据、processor 和参数分组。

## 7. 测试与验收

单元测试采用先失败、后实现的 TDD 顺序，覆盖：

- bbox floor 量化、边界 clamp、非法 bbox；
- manifest `crop_position` 到 scene JSON crop 的精确映射；
- person JSONL schema、ID 唯一性和原子写入；
- reserve 两场景各 500、frame 唯一、固定种子重现；
- replay 进度恢复、重复和空输出拒绝；
- 模型参数互斥分组和学习率；
- DDP world size、梯度累积与 optimizer step 计算；
- checkpoint 数据 hash 不一致时拒绝恢复。

集成验证包括：

1. 对全部 train/dev/test/rl 执行前处理并核对 30,000/2,000/4,328/6,000 条；
2. 使用真实 Florence processor 验证 `<REGION_TO_CATEGORY>` 展开为 `What is the region ...?`；
3. 用小规模 selection 执行真实 base replay 推理；
4. 在可用的 Florence Python 环境中运行单卡 1 optimizer-step smoke test；
5. 如 GPU 和显存允许，再运行 4 卡 DDP 1 optimizer-step smoke test；
6. 在新进程运行 `validate_sft_checkpoint.py`，确认保存模型可重载、processor task 映射未改变。

当前默认 shell 的 Python 环境缺少 torch、transformers、safetensors 和 tqdm，因此实现阶段需要先定位已有 Florence 训练环境；若不存在，则输出明确的依赖文件和环境创建命令。不会在未经确认的情况下启动完整 31,000 条训练。

## 8. 完成标准

满足以下条件视为本阶段完成：

- 四个 person split JSONL 数量和源 manifest 完全一致；
- 1,000 条 replay 均来自 reserve，两个场景各 500，frame 不重复；
- 所有 prompt 使用原生任务 token 和重新量化的合法 loc token；
- 训练入口可用 1 卡调试和 4 卡 torchrun 正式运行；
- 全参数、三组学习率、global batch、累积步数和 scheduler 与设计一致；
- 测试通过，真实 processor prompt 验证通过；
- 可用环境下至少完成 1-step GPU smoke test；
- 原始数据目录和 pretrained checkpoint 文件指纹不变。
