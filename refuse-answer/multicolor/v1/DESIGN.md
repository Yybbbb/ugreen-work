# 杂色检测模块设计文档

## 概述

在颜色分类推理流程中，对分割得到的 upper/lower 区域进行杂色检测。当检测到多色、条纹、方格等情况时，对该区域的颜色分类结果降低置信度。

## 核心方法：方案 A —— 直方图峰值检测 + RLE 纹理分析

### 选择理由

- 速度优先（~5ms/样本），适合批量推理
- 实现简单，参数可解释
- 对大部分纯色样本判断准确

---

## 模块结构

```
segment-color/refuse-answer/multicolor/
├── DESIGN.md                   # 本设计文档
├── multicolor_detector.py      # 核心检测器（对外接口）
├── histogram_peak.py           # 直方图构建 + 峰值检测
├── texture_rle.py              # RLE 游程编码纹理分析
├── analyze_val.py              # 离线批量分析脚本
└── README.md                   # 使用说明
```

---

## 数据流

```
输入: RGB图像 (H,W,3) + 分割mask (H,W) + region_id (upper=1, lower=2)
  │
  ├──[管道 1: 颜色多中心检测]───
  │  1. mask提取像素 → (N, 3) RGB数组
  │  2. RGB → Lab 颜色空间转换
  │  3. 仅取 a*b* 二维 → (N, 2)
  │  4. 128×128 二维直方图
  │  5. 高斯平滑 (σ=2.0, kernel=7×7)
  │  6. 局部极大值检测（8邻域）
  │  7. 低峰过滤 (threshold=0.10 × max_peak)
  │  8. 邻近峰合并 (Δab < 15)
  │  → {n_peaks, peak_positions, peak_heights, purity}
  │
  ├──[管道 2: 纹理分析]───
  │  1. 每个像素分配到最近邻峰值 → 颜色标签图
  │  2. 缩放到 64×64
  │  3. 逐行 RLE 扫描 → 统计每行颜色切换次数
  │  4. 逐列 RLE 扫描 → 统计每列颜色切换次数
  │  5. 计算行/列切换比
  │  6. 纹理类型判定
  │  → {pattern_type, row_switch_ratio, col_switch_ratio}
  │
  └──[汇总]──────────────────
     → {is_multicolor, n_peaks, purity, pattern, confidence_penalty}
```

---

## 算法细节

### 1. RGB → Lab 颜色空间转换

**原因：** RGB 空间中欧氏距离不等于人眼感知距离。Lab 空间中数学距离 ≈ 感知差异。聚类时仅用 `a*b*`（色度），忽略 `L`（明度），避免光照/阴影造成同一颜色被拆分为多类。

**步骤：**

```
1. RGB[0,255] → [0,1] 归一化
2. sRGB Gamma 反变换（线性化）:
     if v ≤ 0.04045: linear = v / 12.92
     else:           linear = ((v + 0.055) / 1.055)^2.4
3. 矩阵变换 RGB_linear → XYZ (D65):
     [X]   [0.4124 0.3576 0.1805] [R_lin]
     [Y] = [0.2126 0.7152 0.0722] [G_lin]
     [Z]   [0.0193 0.1192 0.9505] [B_lin]
4. 白点归一化: X'=X/0.95047, Y'=Y/1.00000, Z'=Z/1.08883
5. 非线性压缩:
     if t > 0.008856: f(t) = t^(1/3)
     else:           f(t) = 7.787×t + 16/116
6. L  = 116×f(Y') - 16
   a* = 500 × (f(X') - f(Y'))
   b* = 200 × (f(Y') - f(Z'))
```

### 2. a\*b\* 二维直方图构建

**参数: N_BINS = 128**

- a\* 和 b\* 各量化为 128 级，覆盖范围 [-128, 127] → bin 宽度 ≈ 2.0 单位
- 结果: 128×128 计数矩阵，每个元素是该 a\*b\* 区间内的像素数
- 复杂度: O(N_pixels)，一次遍历

### 3. 高斯平滑

**参数: σ = 2.0, kernel_size = (7, 7)**

对 128×128 直方图做二维高斯卷积，消除纹理噪声和 JPEG 压缩造成的小毛刺。

```
σ=2.0 → 6σ ≈ 12 bins ≈ 24 ab单位
最小异色间距（红↔橙 Δab≈36）> 24 → 不同颜色不会被平滑合并 ✓
```

### 4. 局部极大值检测

遍历平滑后的直方图，找到所有 8 邻域局部最大值：

```
for each bin (i,j):
    if H_smooth[i,j] > all 8 neighbors:
        peaks.append((i, j, H_smooth[i,j]))
```

### 5. 低峰过滤

**参数: PEAK_THRESHOLD = 0.10**

```
main_peak_height = max(peak_heights)
valid_peaks = [p for p in peaks if p.height >= 0.10 × main_peak_height]
```

次峰面积 < 主峰 10% 则忽略，避免把极小的装饰色块当杂色。

### 6. 邻近峰合并

**参数: PEAK_MERGE_DIST = 15**（a\*b\* 空间欧氏距离）

```
合并规则:
  若两峰在 a*b* 空间的欧氏距离 < 15 → 保留更高的，丢弃较低的
  
阈值推导:
  - 同色光照漂移: Δab 通常 < 10
  - 最小异色间距: 红↔橙 Δab ≈ 36
  - 15 在两者之间，平衡 √
```

### 7. 纯度计算

```
purity = 主峰高度 / 所有峰高度之和

纯色:   purity ≈ 1.0
多色:   purity < 0.85（次峰合计 > 15%）
```

### 8. RLE 纹理分析

**缩放尺寸: 64×64**

#### 8.1 像素量化

每个 ROI 像素 (L, a, b) → 找到最近邻的峰值中心 → 分配颜色标签 (0, 1, ..., K-1)。

#### 8.2 游程编码扫描

```
逐行扫描:
  for each row in 64:
      runs = RLE(row)
      switches_per_row[i] = len(runs) - 1

row_switch_ratio = mean(switches_per_row) / 64

逐列扫描同理 → col_switch_ratio
```

#### 8.3 纹理类型判定

```
if row_switch_ratio < 0.05 and col_switch_ratio < 0.05:
    pattern = "solid"           # 纯色
elif row_switch_ratio >= 0.05 and col_switch_ratio < 0.05:
    pattern = "vertical_stripe"  # 竖条纹（水平方向颜色频繁切换）
elif row_switch_ratio < 0.05 and col_switch_ratio >= 0.05:
    pattern = "horizontal_stripe" # 横条纹（垂直方向颜色频繁切换）
else:
    pattern = "checkered"        # 方格/复杂纹理（双向切换）
```

说明：0.05 的阈值含义是平均每行/列至少有 64×0.05≈3 次颜色切换。低于此值说明颜色在空间上是连片分布的；高于此值说明有交替排列特征。

---

## 最终判定与降权

```python
def classify(n_peaks, purity, pattern):
    if n_peaks <= 1 or purity > 0.85:
        return {
            "is_multicolor": False,
            "confidence_penalty": 1.0    # 不降权
        }
    else:
        if pattern == "solid":  # 多色块拼接（空间上成片但颜色>1）
            return {
                "is_multicolor": True,
                "confidence_penalty": 0.5
            }
        else:  # stripe / checkered — 规则纹理，主色可能仍准确
            return {
                "is_multicolor": True,
                "confidence_penalty": 0.7
            }
```

---

## 输出格式

```json
{
  "is_multicolor": true,
  "n_peaks": 3,
  "peak_colors": [
    {"a": 5, "b": -45, "ratio": 0.55},
    {"a": 60, "b": 40, "ratio": 0.30},
    {"a": -5, "b": 50, "ratio": 0.15}
  ],
  "purity": 0.55,
  "pattern": "checkered",
  "row_switch_ratio": 0.12,
  "col_switch_ratio": 0.09,
  "confidence_penalty": 0.7
}
```

---

## 参数总览

| 参数 | 推荐值 | 作用 | 推导依据 |
|------|--------|------|----------|
| `N_BINS` | 128 | a\*b\* 直方图分辨率 | bin宽≈2.0，光照漂移 < 异色间距 |
| `SIGMA` | 2.0 | 高斯平滑强度 | 6σ=24ab单位 < 最小异色间距36 |
| `KERNEL_SIZE` | (7, 7) | 平滑核尺寸 | 覆盖 3σ 范围 |
| `PEAK_THRESHOLD` | 0.10 | 次峰最低高度比 | 次峰面积 < 10% 不计为杂色 |
| `PEAK_MERGE_DIST` | 15 | 峰合并距离 (Δab) | 在光照漂移上限和异色间距下限之间 |
| `PURITY_THRESHOLD` | 0.85 | 纯色判定线 | 主峰占 85%+ 即视为纯色 |
| `RLE_SIZE` | 64 | 纹理分析缩放尺寸 | 平衡速度和分辨率 |
| `STRIPE_THRESHOLD` | 0.05 | 条纹切换比下限 | 至少 3 次切换/行才考虑条纹 |

---

## 验证计划

1. **峰值检测正确性**：取 20 张已知纯色 + 20 张已知多色样本，人工比对 n_peaks 输出
2. **纹理分析正确性**：分别取纯色/条纹/方格样本各 10 张，验证 pattern 输出
3. **参数敏感性**：对 N_BINS ∈ {64, 128, 256}、PEAK_THRESHOLD ∈ {0.05, 0.10, 0.15} 做网格搜索，找最优组合
4. **与推理流程联调**：在 predict_val_optC.py 后处理中接入，观察降权对颜色分类准确率的影响
