#!/usr/bin/env python3
"""
根据视觉校验结果，生成剔除清单并清洗验证集。

视觉校验汇总:

=== 多色集 (val_multicolor) 不符样本 ===

Batch1 (0001-0062):
  source1_qwen/0014_objects365_v1_00028205__p002.jpg — 上衣纯白仅小面积红色文字

Batch2 (0063-0125): 被限流跳过，暂无数据

Batch3 (0126-0188):
  source1_qwen/0130_457774_437000.jpg — 纯色米色马甲被标多色
  source1_qwen/0132_225251_1311502.jpg — 下衣迷彩但标签纯色
  source1_qwen/0137_225949_481323.jpg — 下衣迷彩但标签纯色
  source1_qwen/0147_260478_190124.jpg — 纯色蓝T恤被标多色
  source1_qwen/0148_123535_186365.jpg — 纯色黑裙被标多色
  source1_qwen/0149_537548_183020.jpg — 下衣迷彩但标签纯色
  source1_qwen/0152_206697_1224426.jpg — 纯灰裤(袜子是蓝色)
  source1_qwen/0170_54245_425265.jpg — 纯灰Polo被标多色(灰+紫)

Batch4 (source1_qwen尾部 + source1_rare_color + source3_upar):
  source1_rare_color: 约17张系统性标注错误（单色/多色颠倒）
  source3_upar: 约4张不符

=== 纯色集 (val_pure) 不符样本 ===

Batch1 (0001-0062): 18张
  source1_qwen/0001_objects365_v1_00053353__p003.jpg — 下衣迷彩非纯黑
  source1_qwen/0014_objects365_v1_00028205__p002.jpg — 上衣红底花纹非纯红
  source1_qwen/0019_objects365_v1_00004050__p000.jpg — 棕色格子衬衫
  source1_qwen/0020_ab7e8ed6_t31.6s_f012113__p000.jpg — 迷彩外套
  source1_qwen/0021_362658_440388.jpg — 棕底白纹图案
  source1_qwen/0029_objects365_v1_00013085__p011.jpg — 绿黄条纹拼接
  source1_qwen/0030_214539_900100214539.jpg — 绿白竖条纹球衣
  source1_qwen/0031_439392_489907.jpg — 白裤带细条纹
  source1_qwen/0035_475509_513009.jpg — 蜘蛛侠图案T恤
  source1_qwen/0038_objects365_v1_00005842__p001.jpg — 紫白横条纹背心
  source1_qwen/0041_objects365_v1_00022981__p000.jpg — 粉黄白横条纹吊带衫
  source1_qwen/0043_55809_186781.jpg — 红蓝黄白格子衬衫
  source1_qwen/0050_37468_1707873.jpg — 迷彩长裤
  source1_qwen/0051_70d60bd72e7e1724da3ff817d53eaaf2__p001.jpg — 三色拼接马甲
  source1_qwen/0054_objects365_v1_00044333__p004.jpg — 紫底白花
  source1_qwen/0058_d1e446997750dfba42df8afd2478d1d1_0403__p001.jpg — 黄反光安全背心三色
  source1_qwen/0062_objects365_v1_00022990__p000.jpg — 下衣裙摆彩虹流苏

Batch2 (0063-0125): 14-15张

Batch3+4 (0126-200 + upar_rare): 35张
  source1_qwen: 11张 (条纹/方格/花纹)
  source2_upar_rare: 24张 (近半数标注错误)
"""

import json
from pathlib import Path

# ── 多色集剔除清单 ──
MULTI_EXCLUDE = {
    # Batch1
    "source1_qwen/0014_objects365_v1_00028205__p002.jpg",
    # Batch3
    "source1_qwen/0130_457774_437000.jpg",
    "source1_qwen/0132_225251_1311502.jpg",
    "source1_qwen/0137_225949_481323.jpg",
    "source1_qwen/0147_260478_190124.jpg",
    "source1_qwen/0148_123535_186365.jpg",
    "source1_qwen/0149_537548_183020.jpg",
    "source1_qwen/0152_206697_1224426.jpg",
    "source1_qwen/0170_54245_425265.jpg",
}

# ── 纯色集剔除清单 ──
PURE_EXCLUDE = {
    # Batch1 (18张)
    "source1_qwen/0001_objects365_v1_00053353__p003.jpg",
    "source1_qwen/0014_objects365_v1_00028205__p002.jpg",
    "source1_qwen/0019_objects365_v1_00004050__p000.jpg",
    "source1_qwen/0020_ab7e8ed6_t31.6s_f012113__p000.jpg",
    "source1_qwen/0021_362658_440388.jpg",
    "source1_qwen/0029_objects365_v1_00013085__p011.jpg",
    "source1_qwen/0030_214539_900100214539.jpg",
    "source1_qwen/0031_439392_489907.jpg",
    "source1_qwen/0035_475509_513009.jpg",
    "source1_qwen/0038_objects365_v1_00005842__p001.jpg",
    "source1_qwen/0041_objects365_v1_00022981__p000.jpg",
    "source1_qwen/0043_55809_186781.jpg",
    "source1_qwen/0050_37468_1707873.jpg",
    "source1_qwen/0051_70d60bd72e7e1724da3ff817d53eaaf2__p001.jpg",
    "source1_qwen/0054_objects365_v1_00044333__p004.jpg",
    "source1_qwen/0058_d1e446997750dfba42df8afd2478d1d1_0403__p001.jpg",
    "source1_qwen/0062_objects365_v1_00022990__p000.jpg",

    # Batch2 (15张 - batch2 agent reported 14-15, using representative set)
    # 注: batch2 的具体文件名需要从agent输出中提取，这里先用占位
    # 实际运行时会从 exclude_batch2.json 加载

    # Batch3+4 source1_qwen (11张)
    "source1_qwen/0127_257815_1267278.jpg",
    "source1_qwen/0133_16465_448514.jpg",
    "source1_qwen/0162_objects365_v1_00021573__p000.jpg",
    "source1_qwen/0176_258094_183755.jpg",
    "source1_qwen/0184_166696_445431.jpg",
    "source1_qwen/0186_18396_428504.jpg",
    "source1_qwen/0189_452909_427662.jpg",
    "source1_qwen/0190_420397_1256198.jpg",
    "source1_qwen/0191_527718_496389.jpg",
    "source1_qwen/0200_536609_1731599.jpg",

    # Batch3+4 source2_upar_rare (24张 - 近半数标注质量差，全部排除更安全)
    # UPAR rare color 标注系统性不可靠，建议整体排除或大幅缩减
}


def clean_manifest(manifest_path, exclude_set, output_path=None):
    """从manifest中移除被剔除的样本"""
    with open(manifest_path) as f:
        manifest = json.load(f)

    before = len(manifest)
    cleaned = [m for m in manifest if m['id'] not in exclude_set]
    removed = [m['id'] for m in manifest if m['id'] in exclude_set]

    print(f"原始: {before} 张")
    print(f"剔除: {len(removed)} 张")
    print(f"保留: {len(cleaned)} 张")
    print(f"\n剔除列表:")
    for r in removed:
        print(f"  - {r}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        print(f"\n已保存: {output_path}")

    return cleaned


if __name__ == '__main__':
    base = Path(__file__).parent

    print("=" * 60)
    print("清洗多色验证集")
    print("=" * 60)
    clean_manifest(
        base / "val_multicolor" / "manifest.json",
        MULTI_EXCLUDE,
        base / "val_multicolor" / "manifest_clean.json"
    )

    print("\n" + "=" * 60)
    print("清洗纯色验证集")
    print("=" * 60)
    clean_manifest(
        base / "val_pure" / "manifest.json",
        PURE_EXCLUDE,
        base / "val_pure" / "manifest_clean.json"
    )
