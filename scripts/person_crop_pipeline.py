#!/usr/bin/env python3
"""
人物检测 + 裁剪脚本（基于 YOLO pose 模型）
===========================================
对一批原始图片做人物检测，裁剪出每个 person 子图，并保存关键点 JSON。

用法:
    python person_crop_pipeline.py \
        --image-dir /path/to/raw/images \
        --output-dir /path/to/person_crops \
        --model yolo26n-pose.pt \
        --conf 0.5 \
        --ext .jpg .png

输出:
    output_dir/
      images/
        00001_{orig_stem}_person_0.jpg   # 裁剪人物图
        00001_{orig_stem}_person_1.jpg
        ...
      annotations/
        00001_{orig_stem}_person_0.json  # bbox + keypoints + 元信息
        ...

依赖:
    pip install ultralytics opencv-python numpy
    # 模型自动下载，或手动放到 --model 路径
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# COCO 17 点关键点名称（与 YOLO pose 输出一致）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 项目的 7 个分割类别
SEG_CLASSES = ["upper", "lower", "dress", "hat", "glasses", "mask", "hair"]


def parse_args():
    p = argparse.ArgumentParser(description="YOLO person detection + cropping")
    p.add_argument("--image-dir", required=True, help="原始图片目录")
    p.add_argument("--output-dir", required=True, help="输出根目录")
    p.add_argument("--model", default="yolo11n-pose.pt",
                   help="YOLO pose 模型路径或名称（默认 yolo11n-pose.pt）")
    p.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    p.add_argument("--max-persons", type=int, default=20, help="每张图最多保留人数")
    p.add_argument("--ext", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"],
                   help="图片扩展名")
    p.add_argument("--margin", type=float, default=0.05,
                   help="裁剪框外扩比例（0.05 = 5%）")
    p.add_argument("--save-viz", action="store_true", help="保存检测可视化图")
    return p.parse_args()


def load_model(model_path: str):
    """加载 YOLO pose 模型（自动下载如果本地没有）。"""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("请先安装 ultralytics: pip install ultralytics")

    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    return model


def find_images(image_dir: Path, extensions: list[str]) -> list[Path]:
    """递归收集所有图片路径。"""
    images = []
    for ext in extensions:
        images.extend(image_dir.rglob(f"*{ext}"))
        images.extend(image_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def run_detection(model, image_path: Path, conf: float, iou: float, max_persons: int):
    """对单张图做 YOLO pose 推理，返回 person 检测列表。"""
    import torch

    results = model(str(image_path), conf=conf, iou=iou, verbose=False)
    result = results[0]

    if result.boxes is None:
        return [], None

    img_h, img_w = result.orig_shape

    # 筛选 person 类 (COCO class_id=0)
    persons = []
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if cls_id != 0:  # 只保留 person
            continue

        score = float(boxes.conf[i].item())
        # bbox 格式: xyxy (绝对像素)
        xyxy = boxes.xyxy[i].cpu().numpy().tolist()
        x1, y1, x2, y2 = [float(v) for v in xyxy]

        # 提取关键点（如果有）
        keypoints_xy = []
        keypoints_conf = []
        if result.keypoints is not None and len(result.keypoints) > i:
            kp = result.keypoints[i]
            if kp.data is not None and len(kp.data) > 0:
                kp_data = kp.data[0].cpu().numpy()
                if kp_data.shape[1] >= 3:
                    keypoints_xy = kp_data[:, :2].tolist()
                    keypoints_conf = kp_data[:, 2].tolist()

        persons.append({
            "person_id": i,
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            "score": score,
            "keypoints_xy": keypoints_xy,
            "keypoints_conf": keypoints_conf,
        })

    # 按分数排序，取 top-k
    persons.sort(key=lambda p: p["score"], reverse=True)
    persons = persons[:max_persons]

    # 可视化（可选）
    viz = None
    if hasattr(result, 'plot'):
        viz = result.plot()

    return persons, viz


def crop_person(image: np.ndarray, bbox_xyxy: list, margin: float) -> np.ndarray:
    """
    按 bbox 裁剪人物子图，带 margin 外扩。

    Args:
        image: BGR 原图 (H, W, 3)
        bbox_xyxy: [x1, y1, x2, y2] 绝对像素坐标
        margin: 外扩比例 (0.05 = 上下左右各扩 5%)

    Returns:
        裁剪后的 BGR 子图
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]

    # 计算外扩
    bw, bh = x2 - x1, y2 - y1
    mx = bw * margin
    my = bh * margin

    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))

    return image[y1:y2, x1:x2]


def build_annotation(source_path: Path, person: dict, crop_img_path: str,
                     crop_w: int, crop_h: int, img_index: int) -> dict:
    """
    构建与现有数据集格式兼容的标注 JSON。
    格式对齐 LIP_clothes_accessory_unified/annotations/ 中的标注结构。
    """
    bbox_xywh = person["bbox_xywh"]
    x, y, w_box, h_box = bbox_xywh

    # 归一化 bbox
    norm_bbox = [x / crop_w, y / crop_h, w_box / crop_w, h_box / crop_h] if crop_w > 0 and crop_h > 0 else None

    return {
        "schema_version": "1.0",
        "sample_id": f"{img_index:05d}_{source_path.stem}_person_{person['person_id']}",
        "source": {
            "name": "custom_dataset",
            "original_path": str(source_path),
            "person_id": person["person_id"],
            "person_bbox_xywh": bbox_xywh,
            "person_score": person["score"],
            "note": "auto-cropped by YOLO pose",
        },
        "image": {
            "path": crop_img_path,
            "width": crop_w,
            "height": crop_h,
            "is_person_crop": True,
        },
        "annotation_path": "",  # 待后续 SAM3 标注流程填充
        "classes": SEG_CLASSES,
        "attributes": {
            cls: {
                "label": "unknown",
                "has_mask": False,
                "bbox": norm_bbox if cls == "upper" else None,
                "mask": None,
                "source_class": cls,
                "score": 0.0,
            }
            for cls in SEG_CLASSES
        },
        "pose": {
            "model": "yolo-pose",
            "generated_at": datetime.now().isoformat(),
            "keypoint_names": KEYPOINT_NAMES,
            "keypoints_xy": person["keypoints_xy"],
            "keypoints_conf": person["keypoints_conf"],
            "image_width": crop_w,
            "image_height": crop_h,
        },
    }


def main():
    args = parse_args()

    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    img_out_dir = output_dir / "images"
    ann_out_dir = output_dir / "annotations"
    viz_out_dir = output_dir / "viz"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    ann_out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_viz:
        viz_out_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(image_dir, args.ext)
    if not images:
        print(f"错误: {image_dir} 下没有找到图片 (ext={args.ext})")
        sys.exit(1)
    print(f"找到 {len(images)} 张图片")

    # 加载模型
    model = load_model(args.model)

    # 逐张处理
    total_persons = 0
    no_person_count = 0
    person_count_list = []
    img_index = 0

    for img_path in images:
        img_index += 1
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [跳过] 无法读取: {img_path}")
            continue

        persons, viz = run_detection(
            model, img_path, args.conf, args.iou, args.max_persons
        )

        if not persons:
            no_person_count += 1
            if img_index % 100 == 0:
                print(f"  [{img_index}/{len(images)}] {img_path.name} → 0 person")
            continue

        person_count_list.append(len(persons))
        orig_h, orig_w = img.shape[:2]

        for person in persons:
            total_persons += 1
            crop = crop_person(img, person["bbox_xyxy"], args.margin)
            crop_h, crop_w = crop.shape[:2]

            # 输出文件名：保持与现有格式兼容
            crop_stem = f"{img_index:05d}_{img_path.stem}_person_{person['person_id']}"
            crop_filename = f"{crop_stem}.jpg"
            ann_filename = f"{crop_stem}.json"

            crop_path = img_out_dir / crop_filename
            ann_path = ann_out_dir / ann_filename

            # 写裁剪图
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 写标注 JSON
            crop_rel = str(crop_path.relative_to(output_dir))
            annotation = build_annotation(
                source_path=img_path,
                person=person,
                crop_img_path=f"images/{crop_filename}",  # 相对路径
                crop_w=crop_w,
                crop_h=crop_h,
                img_index=img_index,
            )
            ann_path.write_text(
                json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        # 保存可视化
        if args.save_viz and viz is not None:
            viz_path = viz_out_dir / f"{img_path.stem}_det.jpg"
            cv2.imwrite(str(viz_path), viz)

        if img_index % 100 == 0:
            avg_p = sum(person_count_list) / len(person_count_list) if person_count_list else 0
            print(f"  [{img_index}/{len(images)}] {img_path.name} → "
                  f"{len(persons)} persons | 累计 {total_persons} 人 | 人均 {avg_p:.1f}")

    # ── 汇总 ──
    avg_persons = sum(person_count_list) / len(person_count_list) if person_count_list else 0
    summary = {
        "pipeline": "yolo_person_crop",
        "model": args.model,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "total_images": len(images),
        "images_with_person": len(images) - no_person_count,
        "images_without_person": no_person_count,
        "total_persons_cropped": total_persons,
        "avg_persons_per_image": round(avg_persons, 2),
    }
    summary_path = output_dir / "person_crop_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  人物裁剪完成！")
    print(f"  总图片: {len(images)}")
    print(f"  检出人物: {total_persons} 人")
    print(f"  无人图片: {no_person_count} 张")
    print(f"  平均人数/图: {avg_persons:.1f}")
    print(f"  裁剪图: {img_out_dir}")
    print(f"  标注:   {ann_out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
