# florence-attibute 数据集与模型产物总结报告

> 本文件是 florence-attibute 仓库中**唯一入库的数据说明**。所有训练数据、预训练权重、SFT/RL checkpoint 均已在 [.gitignore](.gitignore) 中排除,需通过对象存储 / 共享存储单独同步。本报告记录它们的规模、结构、来源与恢复方式,使得在一台干净的机器上 clone 代码后,能够据此重建完整的训练环境。
>
> - 统计时间:2026-08-13
> - 统计口径:`du -sh`(磁盘占用)+ `find -type f | wc -l`(文件数)
> - 仓库总占用:**128 GB**,其中入库内容约 **17 MB / 425 文件**

---

## 1. 总览

| 目录 | 占用 | 占比 | 是否入库 | 性质 |
|------|-----:|-----:|:--------:|------|
| [artifacts/](artifacts/) | 126 GB | 98.4% | 仅实验记录 | SFT / RL 训练产物 |
| [pretrained/](pretrained/) | 888 MB | 0.7% | 否 | Florence-2-base 预训练权重 |
| [data/](data/) | 530 MB | 0.4% | 仅脚本 | 人物裁剪图 + caption 标注 |
| [docs/](docs/) | 12 MB | — | 是 | 实验记录与设计文档 |
| [scripts/](scripts/) | 1.7 MB | — | 是 | 训练 / 评估 / 流水线脚本(27 个) |
| [webdemo/](webdemo/) | 1.4 MB | — | 是 | SFT / RL 结果展示页 |
| [tests/](tests/) | 1.3 MB | — | 是 | 单元测试(24 个) |
| `EXPERIMENT_PLAN.md` | 48 KB | — | 是 | 实验计划 |

单个文件超过 100 MB 的共 **98 个**(GitHub 的硬性拒绝线),全部位于 `artifacts/` 与 `pretrained/`。

---

## 2. 训练数据 `data/` — 530 MB

任务为 Florence-2 原生 `<REGION_TO_CATEGORY>`,即给定人物区域框输出自然语言属性描述,用于后续跨镜人物检索。

### 2.1 划分统计

划分定义见 `data/split_statistics.json`(**已入库**,4 KB):

- 生成时间:2026-08-06
- 来源:`reviewed-person-dataset-merge`
- `caption_similarity_threshold`: 0.9(去重阈值)
- `shortfall_tolerance`: 0.04

| 划分 | 帧数 | 人数 | 已复核 | 占用 | 文件数 |
|------|-----:|-----:|-------:|-----:|-------:|
| train | 15,422 | 30,000 | 1,025 | 178 MB | 15,422 |
| test | 2,432 | 4,328 | 2,912 | 31 MB | 2,432 |
| dev | 944 | 2,000 | 0 | 12 MB | 944 |
| rl | 3,368 | 5,981 | 0 | 37 MB | 3,368 |
| reserve | 6,599 | 21,602 | 0 | 100 MB | 6,599 |

`reserve` 是未投入训练的预留池(21,602 人),主要来自 `tradeshow`(20,992)。测试集的人工复核率最高(2,912 / 4,328 = 67%),训练集仅复核 1,025 条(来自 `小红书` 场景)。

### 2.2 场景构成

共 19 个场景。训练集人数 Top 8:

| 场景 | train | test | rl |
|------|------:|-----:|---:|
| `tradeshow` | 7,403 | 430 | 1,498 |
| `company_surveillance_mp4_adaptive` | 7,205 | 172 | 1,498 |
| `2026_3_20_expo` | 2,447 | 317 | 509 |
| `2026_3_12_europe_city` | 2,129 | 145 | 438 |
| `2026_4_9_ikea` | 2,054 | 154 | 427 |
| `ReID_pedestrian` | 1,930 | 237 | 402 |
| `2026_04_03_cat_cafe` | 1,760 | 321 | 356 |
| `2026_3_17_hardware_store` | 1,174 | 189 | 247 |

场景分布高度不均:`tradeshow` 与 `company_surveillance_mp4_adaptive` 两个场景合计占训练集 48.7%。测试集额外包含 `小红书`(1,025)、`2026_4_17_residential_area`(64)、`2026_4_29_park`(76)三个训练集中不存在或极少的场景,构成一定的分布外检验。

### 2.3 `data/prepared/` — 84 MB / 11 文件

经 `scripts/prepare_person_sft_data.py` 处理后的训练输入。元信息见 `metadata.json`:

- `schema_version`: `florence_person_sft_v1`
- `task`: `<REGION_TO_CATEGORY>`
- 生成时间:2026-08-09

各文件的样本数与 SHA-256(用于校验同步完整性,摘自 `docs/FLORENCE_TRAINING_RUNS.md`):

| 文件 | 样本数 | 占用 | SHA-256 |
|------|-------:|-----:|---------|
| `train.jsonl` | 30,000 | 38 MB | `4752fe4631a27d873bc053caff16ab712b4a09d23ee6ab1ce98fe14ee7d5c575` |
| `test_qwen_attributes.jsonl` | 4,328 | 26 MB | `d95a678714f43c47fad0b8fbbffe84af438cf025d34438acb1a7d5b9d221b0e7` |
| `rl.jsonl` | 5,981 | 7.6 MB | `d0b42e627e4e8913421915a6c27b3706aaf03bc582c5ff1c4fa1d97354859b60` |
| `test.jsonl` | 4,328 | 6.0 MB | `4beeb8dfd29917ffa787a0f84c30f3f3be77d82a1e172d23d61c11c7db82772c` |
| `dev.jsonl` | 2,000 | 2.5 MB | `a725317ca6b3df1a58992975d02b6e9020b6ae9832c22652e23b2f9e3da4b7f9` |
| `native_replay.jsonl` | 1,000 | 1.5 MB | `87b6a1c2a1dc9557283402322e83bf1ec7d71628b6043de5d103db51c49a6900` |

`native_replay.jsonl` 是 1,000 条 Florence 原生 `<REGION_TO_DESCRIPTION>` 样本,用于缓解灾难性遗忘。`test_qwen_attributes.jsonl` 是 Qwen 从 GT caption 抽取的结构化属性,作为当前主评估口径的 GT。

### 2.4 `data/manifests/` — 32 MB / 5 文件

每个划分一份 JSONL,逐样本记录:`sample_id`、`scene`、`session`、`caption`、`crop_index`、`crop_position`、`scale`、`selection_score`、`rl_selection_score`、`source_json_relative_path`。这是从原始帧标注到 prepared 数据的映射索引,**重建 prepared 数据的必要输入**。

### 2.5 `data/review/` — 39 MB / 2,711 文件

人工复核结果,`小红书` 场景占 22 MB。复核后的 caption 由 `data/scripts/rewrite_review_captions.py`(**已入库**)回写。

---

## 3. 预训练权重 `pretrained/Florence-2-base` — 888 MB

**来源**:ModelScope,模型 ID `AI-ModelScope/Florence-2-base`。可直接重新下载,无需纳入版本控制。

| 文件 | 大小 |
|------|-----:|
| `pytorch_model.bin` | 464,421,827 B |
| `model.safetensors` | 463,221,266 B |
| `modeling_florence2.py` | 127 KB |
| `processing_florence2.py` | 49 KB |
| `configuration_florence2.py` | 15 KB |
| `tokenizer.json` / `vocab.json` / 配置若干 | — |

两个权重文件是同一份参数的不同格式,正式训练统一从 `model.safetensors` 加载。离线校验结果(摘自 `EXPERIMENT_PLAN.md`):

| 检查项 | 结果 |
|---|---|
| Processor | `Florence2Processor` |
| Tokenizer | `BartTokenizerFast` |
| 模型类 | `Florence2ForConditionalGeneration` |
| 参数量 | 231,414,016 |
| 权重 dtype | FP16 |

该目录应保持只读原始副本,不在其中修改 processor 或权重。

---

## 4. 训练产物 `artifacts/` — 126 GB

### 4.1 构成

| 子目录 | 占用 | 内容 |
|--------|-----:|------|
| `sft/` | 105 GB | 5 个 SFT run + 1 个历史备份 |
| `rl/` | 22 GB | 2 个 SCST RL run + 评估产物 |
| `reviewed_five_run_pipeline/` | 576 KB | 五轮串行流水线日志 |
| `v4_serial_pipeline/` | 372 KB | V4 串行流水线日志 |

### 4.2 SFT run 清单

五个版本均从 `pretrained/Florence-2-base` 独立初始化,在 GPU 3-6 上串行全参训练。

| 版本 | Run 目录 | 占用 | Global batch | 有效 epoch | Steps |
|------|---------|-----:|-------------:|-----------:|------:|
| V1 | `region_category_person_sft_30k_replay1k_b64` | 11 GB | 64 | 1 | 485 |
| V2 | `region_category_person_sft_30k_replay1k_b32` | 11 GB | 32 | 1 | 969 |
| V3 | `region_category_person_sft_30k_replay1k_b32_e1p5` | 11 GB | 32 | 1.5 | 1,454 |
| V4A | `region_category_person_sft_30k_replay1k_b16_e3_lr125` | 11 GB | 16 | 3 | 5,814 |
| V4B | `region_category_person_sft_30k_person_only_b16_e3_lr125` | 11 GB | 16 | 3 | 5,625 |

另有 `reviewed_retrain_backup_20260809_181345/` **53 GB**,是上述五个 run 的历史备份(5 × 11 GB + `data_prepared` 24 MB),按 `docs/FLORENCE_TRAINING_RUNS.md` 记载为"旧版 run 和旧版 `test_qwen_attributes` 备份"。**这是仓库中最大的单一冗余项,建议确认后清理或归档到冷存储。**

还有 6 个 smoke test 目录(`smoke_one_step_*`、`smoke_stream_batch4*`、`v4_pipeline_smoke_*`),各仅 8 KB。

### 4.3 单个 run 的内部结构

以 V4B 为例,11 GB 由 4 份权重构成:

| 内容 | 占用 |
|------|-----:|
| `final/` | 2.6 GB |
| `checkpoint-5600/` | 2.6 GB |
| `checkpoint-5400/` | 2.6 GB |
| `checkpoint-5200/` | 2.6 GB |
| `test_inference_final/` | 87 MB |
| `dev/` | 3.3 MB |
| `train.log` | 136 KB |
| `run_config.json` | 4 KB |

单个 checkpoint 目录内部:

| 文件 | 大小 | 是否必要 |
|------|-----:|---------|
| `training_state.pt` | 1.85 GB | 仅续训需要(optimizer / scheduler / GradScaler / RNG) |
| `model.safetensors` | 926 MB | **推理必需** |
| `tokenizer.json` | 2.3 MB | 必需 |
| `merges.txt` / `vocab.json` / `special_tokens_map.json` / `added_tokens.json` | ~1.6 MB | 必需 |
| `modeling_florence2.py` / `processing_florence2.py` / `configuration_florence2.py` | 191 KB | 必需 |
| `run_state.json` | 1.6 KB | 记录 |

`training_state.pt` 占单个 checkpoint 的 71%。若不需要续训,**仅同步 `final/` 且剔除 `training_state.pt`,单个模型可压到 930 MB**。

全仓共 21 个 checkpoint 目录(含 `final`),48 个 `.safetensors` + 48 个 `.pt`。

### 4.4 RL run

| Run | 占用 | 说明 |
|-----|-----:|------|
| `region_category_person_scst_reviewed5981_qwen4b` | 11 GB | Qwen4B 语义奖励 |
| `region_category_person_scst_reviewed5981_lexical` | 11 GB | 词汇奖励 |
| `evaluation/` | 193 MB | 三模型对比评估 |
| `evaluation_frozen_20260812` | 191 MB | 2026-08-12 冻结版评估 |
| `evaluation_smoke/` | 228 KB | 冒烟测试 |

两个 RL run 均从 V4B final 初始化,各含 `final/` + `checkpoint-300/400/600`。

`evaluation/` 内部:`qwen_rl` 70 MB、`lexical_rl` 70 MB、`v4b_sft` 54 MB(三组 caption 的推理与抽取产物),以及 `judge_cache.json` 288 KB、`person_caption_metrics.json` 20 KB、`person_caption_comparison.md` 20 KB。**后两个最终报告已入库**。

日志中另有 `*.failed_nccl_step100_20260810.log`,记录了 2026-08-10 在 step 100 处的 NCCL 失败与重启。

### 4.5 已入库的实验记录

`artifacts/` 虽整体排除,但放行了 **329 个小文件 / 约 3.0 MB**:

| 类型 | 说明 |
|------|------|
| `run_config.json` | 完整超参:学习率分层、optimizer、warmup、seed、数据 SHA-256 |
| `train.log` | 训练日志(最大 152 KB) |
| `dev_metrics_step_*.json` | 每 100 step 的 dev 指标(单份约 250 B) |
| `metrics_qwen_gt.json` | 各 run 的最终 Qwen-GT 指标(10 份) |
| `manifest.json` | 测试推理清单 |
| `state.json` / `*.log` | 流水线状态与日志 |
| `person_caption_{metrics.json,comparison.md}` | RL 三模型对比最终报告 |

**放行理由**:这些文件体积极小,但**无法重新生成** —— 重建需要重跑全部 126 GB 的训练(五轮 SFT + 两轮 RL,数十 GPU 时)。它们构成了完整的实验可追溯链条,是权重被剔除后唯一的结果凭据。相对地,`predictions*.jsonl`(单份 2-8 MB)与 `judge_cache.json` 可由 checkpoint 重新推理得到,故排除。

---

## 5. 关键实验结果

完整记录见 [docs/FLORENCE_TRAINING_RUNS.md](docs/FLORENCE_TRAINING_RUNS.md)(已入库)。此处摘录主指标,便于在没有权重的环境下判断该同步哪个 checkpoint。

### 5.1 Qwen-GT 指标(当前主口径)

| 版本 | Micro F1 | Soft Micro F1 | Macro-field F1 | Unknown-extra | 18-24 词 |
|------|---------:|--------------:|---------------:|--------------:|---------:|
| V1 | 0.6451 | 0.7150 | 0.5098 | 14.79% | **59.08%** |
| V2 | 0.6545 | 0.7228 | 0.5230 | 13.84% | 55.91% |
| V3 | 0.6620 | 0.7295 | 0.5324 | 13.03% | 55.22% |
| V4A | 0.6795 | 0.7436 | 0.5573 | 10.99% | 48.59% |
| **V4B** | **0.6812** | **0.7447** | **0.5647** | **10.94%** | 49.12% |

V1→V4B 的 Micro F1 提升 3.61 个百分点,Macro-field F1 提升 5.49 个百分点。**V4B final 是当前属性主指标基线**,也是两个 RL run 的初始化来源 —— 若只同步一个 checkpoint,应选它。

代价是长度合格率:18-24 词比例从 V1 的 59.08% 降到约 49%,平均词数仍在 21.3-21.8,属分布变宽而非整体漂移,故 RL 阶段保留了长度奖励。

### 5.2 训练收敛

| 版本 | 最后 dev step | Dev loss | 训练终点 |
|------|-------------:|---------:|--------:|
| V1 | 450 | 1.6270 | 485 |
| V2 | 900 | 1.5606 | 969 |
| V3 | 1400 | 1.5048 | 1,454 |
| V4A | 5800 | 1.3643 | 5,814 |
| V4B | 5600 | 1.3648 | 5,625 |

### 5.3 主要短板字段

Qwen-GT 逐字段 positive F1 中,V4B 表现最弱的几项:`carried_items.handbag` 0.0954、`handheld_items.dangerous_item` 0.0741、`carried_items.backpack` 0.3416、`head.hairstyle` 0.3685、`handheld_items.mobile_phone` 0.3867。危险物正例极少,非零 F1 仍需扩大样本验证。`head.hair_length` 在 V1 最优(0.8544)并随长训练回退至 0.8351。

---

## 6. 在新机器上重建训练环境

```bash
# 1. clone 代码(约 17 MB,含全部实验记录)
git clone <repo> florence-attibute && cd florence-attibute

# 2. 下载预训练权重(888 MB)
#    ModelScope: AI-ModelScope/Florence-2-base → pretrained/Florence-2-base/
modelscope download --model AI-ModelScope/Florence-2-base --local_dir pretrained/Florence-2-base

# 3. 同步训练数据(530 MB)
rsync -av --progress <shared>/florence-attibute/data/ data/
#    校验 prepared 数据完整性(SHA-256 见本报告 2.3 节)
sha256sum data/prepared/train.jsonl   # 应为 4752fe46...

# 4. 按需拉取 checkpoint —— 不要全量同步
#    仅推理 V4B(推荐,930 MB):
rsync -av --exclude='training_state.pt' \
    <objstore>/artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final/ \
    artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final/
#    需要续训则去掉 --exclude(2.6 GB)

# 5. RL 服务预检(训练前必须通过)
python3 scripts/run_rl_reward_services.py status all
```

### 同步量对比

| 方案 | 体积 |
|------|-----:|
| 全量 | 128 GB |
| 代码 + 数据 + 预训练权重 | 1.4 GB |
| 追加 V4B 推理权重(剔除 optimizer state) | 2.3 GB |
| 追加 V4B + 两个 RL final(均剔除 optimizer state) | 4.2 GB |

即**常规复现只需 1.4 GB,含最优模型推理为 2.3 GB**,相比全量减少 98%。

### 迁移注意事项

1. **绝对路径** —— `data/prepared/metadata.json`、各 `run_config.json` 中的 `data_root`、`model_source`、`files.*.path` 均为 `/data1/work/MichaelYu/florence-attibute/...` 绝对路径,新环境需改写或对齐目录结构。
2. **RL 依赖外部推理服务** —— 训练与评估依赖两个固定 vLLM 服务:GPU 1 的 `Qwen3.6-27B-FP8`(端口 6097,vLLM 0.19.1,context 65,536,并发 64,强制 FP8 Marlin)与 GPU 2 的 `Qwen3.5-4B`(端口 6098,context 32,768,并发 128)。两者均为纯文本、默认关闭 thinking。这些服务不在本仓库范围内,需单独部署。
3. **续训的一致性约束** —— `--resume-from` 必须恢复 optimizer、scheduler、GradScaler、epoch/batch/global-step、top-3 和每 rank 的 RNG。任何数据、alpha、生成参数或 reward 配置不一致都会被拒绝续训,因此 `training_state.pt` 一旦剔除即无法续训,只能重新开始。
4. **硬件基线** —— 4 × RTX 6000 Ada(GPU 3-6),BF16 AMP + FP32 optimizer state。

---

## 7. 排除规则说明

[.gitignore](.gitignore) 采用"先全排除,再放行小记录"的策略:

| 规则 | 排除内容 | 理由 |
|------|---------|------|
| `artifacts/**` | 126 GB | 训练产物整体排除 |
| `!artifacts/**/` | — | 允许 git 遍历子目录,否则放行规则不生效 |
| `!artifacts/**/{run_config,state,manifest,metrics_qwen_gt}.json` | — | 放行实验配置与指标 |
| `!artifacts/**/*.log` | — | 放行训练日志 |
| `!artifacts/**/dev_metrics_step_*.json` | — | 放行 dev 指标 |
| `*.safetensors` `*.pt` `*.bin` | 权重与 optimizer state | 即使在放行目录内也排除 |
| `artifacts/**/*.jsonl` | 预测结果 | 可由 checkpoint 重新推理 |
| `artifacts/**/judge_cache.json` | 288 KB | Qwen 判定缓存,可重新生成 |
| `pretrained/` | 888 MB | 从 ModelScope 下载 |
| `data/*` + `!data/scripts/` + `!data/split_statistics.json` | 530 MB | 数据走共享存储,保留脚本与划分统计 |
| `!DATASET_REPORT.md` | — | 确保本报告入库 |

`*.safetensors` / `*.pt` / `*.bin` 三条规则置于放行规则之后,是为了防止 `!artifacts/**/manifest.json` 一类的目录放行意外带出同目录的权重文件 —— git 的忽略规则按顺序匹配、后者覆盖前者。

**入库结果**:17 MB / 425 文件,其中 12 MB 是 `docs/florence_person_attribute_experiment_0730.html`(实验可视化报告)。若要进一步压缩仓库,这个 HTML 是首选目标。`git status` 在 128 GB 的目录树上耗时 5 ms,说明排除规则未触发对权重目录的遍历。
