# Florence2 SFT Batch 与流式数据加载实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Florence2-base 四卡全参数 SFT 调整为 global batch 64 和更低学习率，并让训练按 JSONL 字节偏移流式读取已预处理样本。

**Architecture:** `data/prepared/*.jsonl` 保持为离线、可审计的 Florence 样本 manifest，其中已经包含 prompt、bbox、label 和图像路径。训练启动时顺序扫描 JSONL 建立轻量字节偏移索引，DDP sampler 打乱索引并分发给各 rank；DataLoader worker 只读取当前样本，collator 只处理当前 batch 的图像与文本。

**Tech Stack:** Python 3、PyTorch DDP/DataLoader、Transformers Florence2 processor、JSONL、pytest。

---

### Task 1: 锁定新的默认训练配置

**Files:**
- Modify: `tests/test_train_person_attribute_sft.py`
- Modify: `scripts/train_person_attribute_sft.py`

- [x] **Step 1: 写默认参数失败测试**

断言 parser 默认值为每卡 batch 4、累积 4、视觉/投影/语言学习率分别为 `1e-7`、`5e-7`、`1e-6`，warmup 为 `0.05`，4 卡总步数为 485。

- [x] **Step 2: 运行测试并确认失败**

Run: `python -m pytest -q tests/test_train_person_attribute_sft.py`

Expected: FAIL，显示旧默认值 batch 2、旧学习率或旧 warmup。

- [x] **Step 3: 修改 parser 默认值**

只修改上述已确认超参数，不改变命令行覆盖行为。

- [x] **Step 4: 运行测试并确认通过**

Run: `python -m pytest -q tests/test_train_person_attribute_sft.py`

Expected: PASS。

### Task 2: 用字节偏移替代全量 JSON 对象加载

**Files:**
- Modify: `tests/test_train_person_attribute_sft.py`
- Modify: `scripts/train_person_attribute_sft.py`

- [x] **Step 1: 写流式索引失败测试**

测试临时 JSONL 的顺序扫描、按索引随机访问、多文件拼接、数量限制、重复 `sample_id` 拒绝，以及索引对象不保存完整 row 字典。

- [x] **Step 2: 运行目标测试并确认失败**

Run: `python -m pytest -q tests/test_train_person_attribute_sft.py -k indexed`

Expected: FAIL，因为流式索引接口尚不存在。

- [x] **Step 3: 实现 `IndexedJsonlRows`**

以二进制模式记录 `(path, byte_offset, expected_task)`；扫描时逐行验证 schema 和唯一 ID，随后释放临时 ID set。`__getitem__` 在当前 worker 中懒加载文件句柄、seek 到 offset、解析单行 JSON 并再次执行轻量校验。

- [x] **Step 4: 接入训练与 dev DataLoader**

训练数据由 30,000 person 与 1,000 replay 两个索引拼接，dev 使用 2,000 person 索引。移除训练开始时创建完整 row list 和 Python 级预 shuffle，统一由 `DistributedSampler(shuffle=True, seed=20260720)` 打乱；单卡也使用 seeded sampler，保证行为一致。

- [x] **Step 5: 运行流式数据测试并确认通过**

Run: `python -m pytest -q tests/test_train_person_attribute_sft.py -k 'indexed or streaming or optimizer_steps'`

Expected: PASS。

### Task 3: 更新实验计划

**Files:**
- Modify: `EXPERIMENT_PLAN.md`

- [x] **Step 1: 更新第 4 章数据流**

说明离线 JSONL 只保存元数据，训练时按 offset 读取单条记录、按 batch 解码图像和调用 processor，不预存 pixel tensor；DDP sampler 每 epoch 打乱并按 rank 分片。

- [x] **Step 2: 更新第 9 章 SFT 超参数**

记录 4 卡、每卡 batch 4、累积 4、global batch 64、约 485 step、三组低学习率、5% warmup、每 50 step 评测，以及 batch 4 OOM 时 batch 2/累积 8 的等效回退配置。

- [x] **Step 3: 增加正式训练前门槛**

先用 32 条训练样本做四卡 smoke，检查每个 rank 有进程、峰值显存、非有限 loss、吞吐和数据 worker 稳定性；不通过则不启动完整训练。

### Task 4: 完整验证

**Files:**
- Verify: `tests/test_train_person_attribute_sft.py`
- Verify: `tests/test_prepare_person_sft_data.py`
- Verify: `EXPERIMENT_PLAN.md`

- [x] **Step 1: 运行完整训练相关单测**

Run: `python -m pytest -q tests/test_train_person_attribute_sft.py tests/test_prepare_person_sft_data.py tests/test_person_sft_data.py`

Expected: 全部 PASS。

- [x] **Step 2: 编译检查**

Run: `python -m py_compile scripts/train_person_attribute_sft.py`

Expected: exit 0。

- [x] **Step 3: 校验正式数据而不启动训练**

Run: `python scripts/train_person_attribute_sft.py --validate-data-only`

Expected: 识别 30,000 person、1,000 replay、2,000 dev，进程内只保留索引。

- [x] **Step 4: 运行受限 smoke**

仅在四张卡空闲时用 `--max-train-samples 32 --max-dev-samples 16 --max-optimizer-steps 1 --no-save --allow-nonstandard-counts` 启动 4 卡；若四卡不可用，只完成 CPU 数据流校验并明确报告未执行 GPU smoke。
