#!/usr/bin/env python3
"""
从 extract_frames 的所有帧中检测并裁剪人物图。

输入: extradata/extract_frames/{YYYY-MM-DD_HH-MM-SS}/000001.jpg ...
输出: extradata/person_crops/
        images/    {YYYY-MM-DD_HH-MM-SS}_{frame}_person_{id}.jpg
        annotations/{YYYY-MM-DD_HH-MM-SS}_{frame}_person_{id}.json

用法:
    cd /data1/work/MichaelYu/segment-color
    pip install ultralytics opencv-python  # 首次
    python scripts/crop_persons_from_frames.py --conf 0.5
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FRAMES_DIR = PROJECT_ROOT / "extradata" / "extract_frames"
OUTPUT_DIR = PROJECT_ROOT / "extradata" / "person_crops"

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SEG_CLASSES = ["upper", "lower", "dress", "hat", "glasses", "mask", "hair"]


def parse_args():
    p = argparse.ArgumentParser(description="从 extract_frames 裁剪人物")
    p.add_argument("--frames-dir", default=str(FRAMES_DIR))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    p.add_argument("--margin", type=float, default=0.05, help="裁剪框外扩比例")
    p.add_argument("--max-persons", type=int, default=20, help="每帧最多保留人数")
    p.add_argument("--step", type=int, default=2,
                   help="帧间隔：step=2 则只取 000001, 000003, 000005...（隔1取1）")
    return p.parse_args()


def collect_images(frames_dir: Path, step: int = 2) -> list[Path]:
    """收集图片，按帧号隔 step 取一张（step=2 则取 000001, 000003, ...）。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    skipped = 0
    for subdir in sorted(frames_dir.iterdir()):
        if subdir.is_dir():
            for img in sorted(subdir.iterdir()):
                if not (img.is_file() and img.suffix.lower() in exts):
                    continue
                # 解析帧号，只取奇数帧（step=2 时 000001 取, 000002 跳过）
                try:
                    frame_num = int(img.stem)
                except ValueError:
                    images.append(img)  # 非纯数字文件名保留
                    continue
                if (frame_num - 1) % step == 0:
                    images.append(img)
                else:
                    skipped += 1
    if skipped:
        print(f"[step={step}] 跳过 {skipped} 张相邻帧")
    return images


def detect_persons(model, image_path: Path, conf: float, iou: float) -> list[dict]:
    """YOLO pose 推理，返回该帧中所有 person 的检测结果。"""
    results = model(str(image_path), conf=conf, iou=iou, verbose=False)
    result = results[0]

    if result.boxes is None:
        return []

    persons = []
    for i in range(len(result.boxes)):
        if int(result.boxes.cls[i].item()) != 0:  # 仅 person 类
            continue

        score = float(result.boxes.conf[i].item())
        x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().tolist()

        kp_xy, kp_conf = [], []
        if result.keypoints is not None and i < len(result.keypoints):
            kp = result.keypoints[i]
            if kp.data is not None and len(kp.data) > 0:
                arr = kp.data[0].cpu().numpy()
                if arr.shape[1] >= 3:
                    kp_xy = arr[:, :2].tolist()
                    kp_conf = arr[:, 2].tolist()

        persons.append({
            "score": score,
            "bbox_xyxy": [x1, y1, x2, y2],
            "keypoints_xy": kp_xy,
            "keypoints_conf": kp_conf,
        })

    persons.sort(key=lambda p: p["score"], reverse=True)
    return persons


def crop_person(image, bbox_xyxy: list, margin: float, save_path: Path) -> tuple[int, int]:
    """按 bbox 裁剪并保存为 JPEG，返回 (宽, 高)。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy

    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin

    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))

    crop = image[y1:y2, x1:x2]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return crop.shape[1], crop.shape[0]


def build_stem(frame_path: Path, person_idx: int) -> str:
    """
    命名: {时间戳目录}_{帧号}_person_{序号}
    例: 2026-03-24_06-08-59_000001_person_0
    """
    ts = frame_path.parent.name    # 2026-03-24_06-08-59
    fn = frame_path.stem           # 000001
    return f"{ts}_{fn}_person_{person_idx}"


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)

    if not frames_dir.exists():
        sys.exit(f"目录不存在: {frames_dir}")

    images = collect_images(frames_dir, step=args.step)
    if not images:
        sys.exit(f"未找到图片于 {frames_dir}")
    print(f"共 {len(images)} 张帧图片\n")

    # 加载 YOLO（自动下载模型）
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")
    print("模型就绪\n")

    img_dir = output_dir / "images"
    ann_dir = output_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    total_persons = 0
    no_person = 0
    t0 = time.time()

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        persons = detect_persons(model, img_path, args.conf, args.iou)

        if not persons:
            no_person += 1
        else:
            for pi, p in enumerate(persons[:args.max_persons]):
                total_persons += 1
                stem = build_stem(img_path, pi)

                # 保存裁剪图
                crop_w, crop_h = crop_person(img, p["bbox_xyxy"], args.margin,
                                             img_dir / f"{stem}.jpg")

                # 保存标注 JSON
                ann = {
                    "schema_version": "1.0",
                    "sample_id": stem,
                    "source": {
                        "name": "extradata_extract_frames",
                        "original_path": str(img_path),
                        "timestamp_dir": img_path.parent.name,
                        "frame": img_path.stem,
                        "person_id": pi,
                        "person_score": p["score"],
                        "person_bbox_xyxy": p["bbox_xyxy"],
                    },
                    "image": {
                        "path": f"images/{stem}.jpg",
                        "width": crop_w,
                        "height": crop_h,
                        "is_person_crop": True,
                    },
                    "classes": SEG_CLASSES,
                    "attributes": {
                        c: {"label": "unknown", "has_mask": False, "bbox": None, "mask": None}
                        for c in SEG_CLASSES
                    },
                    "pose": {
                        "model": "yolo11n-pose.pt",
                        "generated_at": datetime.now().isoformat(),
                        "keypoint_names": KEYPOINT_NAMES,
                        "keypoints_xy": p["keypoints_xy"],
                        "keypoints_conf": p["keypoints_conf"],
                    },
                }
                (ann_dir / f"{stem}.json").write_text(
                    json.dumps(ann, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

        if idx % 500 == 0:
            elapsed = time.time() - t0
            speed = idx / max(elapsed, 0.1)
            eta_min = (len(images) - idx) / max(speed, 0.01) / 60
            print(f"  [{idx}/{len(images)}] {img_path.parent.name}/{img_path.name}  "
                  f"| {speed:.0f} fps | 累计人: {total_persons} | ETA {eta_min:.1f} min")

    # ── 汇总 ──
    elapsed = time.time() - t0
    summary = {
        "total_frames": len(images),
        "frames_no_person": no_person,
        "total_persons": total_persons,
        "elapsed_min": round(elapsed / 60, 1),
        "fps": round(len(images) / max(elapsed, 0.1), 1),
    }
    (output_dir / "person_crop_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print(f"  帧总数:      {len(images)}")
    print(f"  裁剪人物:    {total_persons}")
    print(f"  无人物帧:    {no_person}")
    print(f"  耗时:        {elapsed/60:.1f} 分钟 ({elapsed:.0f}s)")
    print(f"  裁剪图目录:  {img_dir}")
    print(f"  标注目录:    {ann_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
