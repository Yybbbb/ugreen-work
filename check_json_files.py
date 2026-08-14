#!/usr/bin/env python3
"""检查标注文件是否都存在且有合法的JSON内容"""

import json
import os
from pathlib import Path

def check_json_files(txt_file, base_dir):
    """
    检查txt文件中第三列的所有json文件
    """
    missing_files = []
    invalid_json_files = []
    valid_count = 0
    total_count = 0

    print(f"读取文件: {txt_file}")
    print(f"基础路径: {base_dir}")
    print("-" * 80)

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                print(f"⚠️  行 {line_num}: 格式不正确，列数不足 - {line}")
                continue

            # 第三列是color-label json路径（索引为2）
            json_rel_path = parts[2]
            json_full_path = os.path.join(base_dir, json_rel_path)

            total_count += 1

            # 检查文件是否存在
            if not os.path.exists(json_full_path):
                missing_files.append({
                    'line': line_num,
                    'path': json_rel_path,
                    'full_path': json_full_path
                })
                continue

            # 检查JSON是否合法
            try:
                with open(json_full_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)

                # 检查是否为空或无效
                if data is None:
                    invalid_json_files.append({
                        'line': line_num,
                        'path': json_rel_path,
                        'reason': 'JSON内容为null'
                    })
                elif isinstance(data, dict) and len(data) == 0:
                    invalid_json_files.append({
                        'line': line_num,
                        'path': json_rel_path,
                        'reason': 'JSON为空字典'
                    })
                else:
                    valid_count += 1

            except json.JSONDecodeError as e:
                invalid_json_files.append({
                    'line': line_num,
                    'path': json_rel_path,
                    'reason': f'JSON解析错误: {str(e)}'
                })
            except Exception as e:
                invalid_json_files.append({
                    'line': line_num,
                    'path': json_rel_path,
                    'reason': f'读取错误: {str(e)}'
                })

    # 打印结果
    print(f"\n📊 检查统计:")
    print(f"  总文件数: {total_count}")
    print(f"  ✅ 有效文件: {valid_count}")
    print(f"  ❌ 缺失文件: {len(missing_files)}")
    print(f"  ⚠️  无效JSON: {len(invalid_json_files)}")
    print()

    # 打印缺失的文件
    if missing_files:
        print(f"❌ 缺失的文件 ({len(missing_files)} 个):")
        print("-" * 80)
        for item in missing_files[:20]:  # 只显示前20个
            print(f"  行 {item['line']}: {item['path']}")
        if len(missing_files) > 20:
            print(f"  ... 还有 {len(missing_files) - 20} 个缺失文件")
        print()

    # 打印无效的JSON文件
    if invalid_json_files:
        print(f"⚠️  无效的JSON文件 ({len(invalid_json_files)} 个):")
        print("-" * 80)
        for item in invalid_json_files[:20]:  # 只显示前20个
            print(f"  行 {item['line']}: {item['path']}")
            print(f"    原因: {item['reason']}")
        if len(invalid_json_files) > 20:
            print(f"  ... 还有 {len(invalid_json_files) - 20} 个无效文件")
        print()

    # 总结
    if not missing_files and not invalid_json_files:
        print("✅ 所有文件都存在且包含合法的JSON内容！")
    else:
        print("⚠️  发现问题，请检查上述列表")

    return {
        'total': total_count,
        'valid': valid_count,
        'missing': missing_files,
        'invalid': invalid_json_files
    }

if __name__ == '__main__':
    txt_file = '/data1/work/MichaelYu/segment-color/data/object_detection_0309-0429_random200_color_label.txt'
    base_dir = '/data1/work/MichaelYu/segment-color/data'

    result = check_json_files(txt_file, base_dir)
