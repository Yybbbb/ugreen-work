#!/usr/bin/env python3
"""
IPC 训练集全量标注脚本 (多服务并发版)
========================================
对每张图片检测所有目标并标注属性，输出单个 JSON:
  - person + 属性 (top_color, gender, age_group, mask_wearing)
  - vehicle + 颜色
  - cat / dog / package (仅 bbox)

mask_wearing 使用分割 mask 包含判定，其他属性使用 IoU 匹配。

用法:
    python annotate_all_train.py --workers 20 --resume
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    sys.exit("请先安装 requests: pip install requests")

try:
    from pycocotools import mask as mask_utils
except ImportError:
    sys.exit("请先安装 pycocotools: pip install pycocotools")

try:
    from tqdm import tqdm
except ImportError:
    print("提示: pip install tqdm 可显示进度条")
    tqdm = None


class ImageTimeoutError(Exception):
    """单个请求超时，跳过整张图片。"""
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# SAM3 多实例（如 server_batch / manage.sh）所在机器；端口与启动脚本一致时可不改
SAM3_SERVER_HOST = "192.168.111.2"
SAM3_PORT_BASE = 8020
SAM3_NUM_SERVER_INSTANCES = 14  # 8020..8033；实例数不同请改此处或运行时 --servers

DEFAULT_SERVERS = [
    *(
        f"http://{SAM3_SERVER_HOST}:{SAM3_PORT_BASE + i}"
        for i in range(SAM3_NUM_SERVER_INSTANCES)
    ),
]

IMAGE_DIR = "/data/work/XuDePeng/data0225_ipc/images/train"
OUTPUT_DIR = "/data/work/LuoWeiXing/ipc_dataset_sam3_annotation/output_train_all"

PERSON_CONFIDENCE = 0.7
VEHICLE_CONFIDENCE = 0.5
ATTR_CONFIDENCE = 0.5
SIMPLE_CONFIDENCE = 0.5
IOU_THRESHOLD = 0.3
MASK_CONTAINMENT_THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

PERSON_ATTR_CONFIG = {
    "top_color": {
        "default": "unknown",
        "prompts": {
            "black":      "person in black top",
            "white":      "person in white top",
            "red":        "person in red top",
            "yellow":     "person in yellow top",
            "green":      "person in green top",
            "blue":       "person in blue top",
            "gray":       "person in gray top",
            "multicolor": "person in multicolor top",
        },
    },
    "gender": {
        "default": "unknown",
        "prompts": {
            "male":   "man",
            "female": "woman",
        },
    },
    "age_group": {
        "default": "unknown",
        "prompts": {
            "adult":   "adult",
            "child":   "child",
            "elderly": "elderly person",
        },
    },
}

VEHICLE_COLOR_CONFIG = {
    "default": "unknown",
    "prompts": {
        "black":      "black car",
        "white":      "white car",
        "gray":       "gray car",
        "red":        "red car",
        "yellow":     "yellow car",
        "green":      "green car",
        "blue":       "blue car",
        "multicolor": "multicolor car",
    },
}

SIMPLE_OBJECTS = ["cat", "dog", "package"]


def build_all_prompts():
    """Return list of (tag, prompt_text, confidence, needs_mask)."""
    prompts = []
    prompts.append(("person", "person", PERSON_CONFIDENCE, True))
    prompts.append(("face_mask", "mask", ATTR_CONFIDENCE, True))
    for attr_name, cfg in PERSON_ATTR_CONFIG.items():
        for attr_value, prompt_text in cfg["prompts"].items():
            prompts.append((f"person_attr:{attr_name}:{attr_value}", prompt_text, ATTR_CONFIDENCE, False))
    prompts.append(("vehicle", "vehicle", VEHICLE_CONFIDENCE, False))
    for color_value, prompt_text in VEHICLE_COLOR_CONFIG["prompts"].items():
        prompts.append((f"vehicle_color:{color_value}", prompt_text, ATTR_CONFIDENCE, False))
    for obj in SIMPLE_OBJECTS:
        prompts.append((f"simple:{obj}", obj, SIMPLE_CONFIDENCE, False))
    return prompts

ALL_PROMPTS = build_all_prompts()


def parse_args():
    p = argparse.ArgumentParser(description="IPC Train 全量标注")
    p.add_argument("--servers", default=None, help="SAM3 服务地址列表，逗号分隔")
    p.add_argument("--image-dir", default=IMAGE_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    p.add_argument("--mask-containment-threshold", type=float, default=MASK_CONTAINMENT_THRESHOLD)
    p.add_argument("--timeout", type=int, default=20, help="单个请求超时秒数，超时则跳过该图片")
    p.add_argument("--workers", type=int, default=10, help="并发图片数")
    p.add_argument("--resume", action="store_true", help="断点续跑")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Round-robin server pool
# ---------------------------------------------------------------------------

class ServerPool:
    def __init__(self, servers, max_concurrent_per_server=1):
        self.servers = list(servers)
        self._idx = 0
        self._lock = threading.Lock()
        self._semaphore = None
        self._max_per_server = max_concurrent_per_server

    def init_semaphore(self):
        limit = len(self.servers) * self._max_per_server
        self._semaphore = threading.Semaphore(limit)
        print(f"  全局并发上限: {limit} (服务数 {len(self.servers)} × {self._max_per_server})")

    def next(self):
        with self._lock:
            s = self.servers[self._idx % len(self.servers)]
            self._idx += 1
            return s

    def acquire(self):
        if self._semaphore:
            self._semaphore.acquire()

    def release(self):
        if self._semaphore:
            self._semaphore.release()

    def check_health(self, timeout=5):
        alive = []
        for s in self.servers:
            try:
                r = requests.get(f"{s.rstrip('/')}/health", timeout=timeout)
                r.raise_for_status()
                alive.append(s)
                print(f"  [ok]   {s}")
            except Exception:
                print(f"  [FAIL] {s}")
        return alive


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def compute_iou(box_a, box_b):
    a = xywh_to_xyxy(box_a)
    b = xywh_to_xyxy(box_b)
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def decode_rle_mask(rle_str, h, w):
    """Decode a COCO-style RLE string into a binary numpy mask."""
    rle = {"size": [h, w], "counts": rle_str.encode("utf-8") if isinstance(rle_str, str) else rle_str}
    return mask_utils.decode(rle)


def compute_containment(small_mask, big_mask):
    """Fraction of small_mask pixels that fall inside big_mask."""
    small_area = small_mask.sum()
    if small_area == 0:
        return 0.0
    intersection = (small_mask & big_mask).sum()
    return float(intersection) / float(small_area)


# ---------------------------------------------------------------------------
# SAM3 API
# ---------------------------------------------------------------------------

def call_annotate(server, image_path, prompt, confidence_threshold, timeout):
    url = f"{server.rstrip('/')}/annotate"
    with open(image_path, "rb") as f:
        resp = requests.post(
            url,
            files={"image": (os.path.basename(image_path), f)},
            data={"prompt": prompt, "confidence_threshold": str(confidence_threshold)},
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Per-image annotation
# ---------------------------------------------------------------------------

def annotate_image(server, image_path, iou_threshold, mask_containment_threshold, timeout):
    """Send all prompts sequentially to a single dedicated server."""
    t0 = time.time()
    total_requests = 0
    results = {}

    for tag, prompt_text, confidence, needs_mask in ALL_PROMPTS:
        total_requests += 1
        try:
            result = call_annotate(server, image_path, prompt_text, confidence, timeout)
            results[tag] = result
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            raise ImageTimeoutError(
                f"Request timeout/connection error after {timeout}s on prompt '{prompt_text}'"
            )
        except Exception:
            continue

    # --- Unpack base detections ---
    person_result = results.get("person", {})
    img_h = person_result.get("orig_img_h", 0)
    img_w = person_result.get("orig_img_w", 0)

    # If person result is empty, try to get image dimensions from any result
    if not img_h or not img_w:
        for r in results.values():
            if r.get("orig_img_h"):
                img_h = r["orig_img_h"]
                img_w = r["orig_img_w"]
                break

    # --- Build person records ---
    person_boxes = person_result.get("pred_boxes", [])
    person_scores = person_result.get("pred_scores", [])
    person_masks_rle = person_result.get("pred_masks", [])

    persons = []
    for i in range(len(person_boxes)):
        persons.append({
            "person_id": i,
            "bbox": person_boxes[i],
            "person_score": person_scores[i] if i < len(person_scores) else 0.0,
            "top_color": "unknown", "top_color_score": 0.0,
            "color_details": [],
            "gender": "unknown", "gender_score": 0.0,
            "age_group": "unknown", "age_group_score": 0.0,
            "mask_wearing": "no", "mask_wearing_score": 0.0,
        })

    # --- Person attribute matching (IoU) ---
    for tag, result in results.items():
        if not tag.startswith("person_attr:"):
            continue
        if result.get("status") == "error" or not result.get("pred_boxes"):
            continue
        _, attr_name, attr_value = tag.split(":", 2)
        attr_boxes = result["pred_boxes"]
        attr_scores = result.get("pred_scores", [0.0] * len(attr_boxes))
        attr_masks = result.get("pred_masks", [])

        for j, a_box in enumerate(attr_boxes):
            a_score = attr_scores[j] if j < len(attr_scores) else 0.0
            best_idx, best_iou = -1, 0.0
            for pi in range(len(person_boxes)):
                iou = compute_iou(a_box, person_boxes[pi])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = pi
            if best_idx >= 0 and best_iou >= iou_threshold:
                score_key = f"{attr_name}_score"
                if a_score > persons[best_idx][score_key]:
                    persons[best_idx][attr_name] = attr_value
                    persons[best_idx][score_key] = a_score
                if attr_name == "top_color":
                    detail = {"color": attr_value, "score": a_score, "bbox": a_box}
                    if j < len(attr_masks):
                        detail["mask"] = attr_masks[j]
                    persons[best_idx]["color_details"].append(detail)

    # --- Mask wearing via mask containment ---
    face_mask_result = results.get("face_mask", {})
    if (face_mask_result.get("pred_masks")
            and person_masks_rle
            and img_h > 0 and img_w > 0):
        face_mask_boxes = face_mask_result.get("pred_boxes", [])
        face_mask_scores = face_mask_result.get("pred_scores", [])
        face_mask_rles = face_mask_result["pred_masks"]

        person_binary_masks = [decode_rle_mask(rle, img_h, img_w) for rle in person_masks_rle]
        face_binary_masks = [decode_rle_mask(rle, img_h, img_w) for rle in face_mask_rles]

        for fm_idx, fm_mask in enumerate(face_binary_masks):
            fm_score = face_mask_scores[fm_idx] if fm_idx < len(face_mask_scores) else 0.0
            best_person_idx = -1
            best_containment = 0.0
            for pi, p_mask in enumerate(person_binary_masks):
                containment = compute_containment(fm_mask, p_mask)
                if containment > best_containment:
                    best_containment = containment
                    best_person_idx = pi
            if best_person_idx >= 0 and best_containment >= mask_containment_threshold:
                if fm_score > persons[best_person_idx]["mask_wearing_score"]:
                    persons[best_person_idx]["mask_wearing"] = "yes"
                    persons[best_person_idx]["mask_wearing_score"] = fm_score

    # --- Build vehicle records ---
    vehicle_result = results.get("vehicle", {})
    vehicle_boxes = vehicle_result.get("pred_boxes", [])
    vehicle_scores = vehicle_result.get("pred_scores", [])

    vehicles = []
    for i in range(len(vehicle_boxes)):
        vehicles.append({
            "vehicle_id": i,
            "bbox": vehicle_boxes[i],
            "vehicle_score": vehicle_scores[i] if i < len(vehicle_scores) else 0.0,
            "color": "unknown", "color_score": 0.0,
            "color_details": [],
        })

    for tag, result in results.items():
        if not tag.startswith("vehicle_color:"):
            continue
        if result.get("status") == "error" or not result.get("pred_boxes"):
            continue
        color_value = tag.split(":", 1)[1]
        color_boxes = result["pred_boxes"]
        color_scores = result.get("pred_scores", [0.0] * len(color_boxes))
        color_masks = result.get("pred_masks", [])

        for j, c_box in enumerate(color_boxes):
            c_score = color_scores[j] if j < len(color_scores) else 0.0
            best_idx, best_iou = -1, 0.0
            for vi in range(len(vehicle_boxes)):
                iou = compute_iou(c_box, vehicle_boxes[vi])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = vi
            if best_idx >= 0 and best_iou >= iou_threshold:
                if c_score > vehicles[best_idx]["color_score"]:
                    vehicles[best_idx]["color"] = color_value
                    vehicles[best_idx]["color_score"] = c_score
                detail = {"color": color_value, "score": c_score, "bbox": c_box}
                if j < len(color_masks):
                    detail["mask"] = color_masks[j]
                vehicles[best_idx]["color_details"].append(detail)

    # --- Simple objects ---
    simple_objects = {}
    for obj_name in SIMPLE_OBJECTS:
        tag = f"simple:{obj_name}"
        result = results.get(tag, {})
        items = []
        if result.get("pred_boxes"):
            for i, box in enumerate(result["pred_boxes"]):
                score = result["pred_scores"][i] if i < len(result.get("pred_scores", [])) else 0.0
                items.append({"bbox": box, "score": score})
        simple_objects[obj_name + "s"] = items

    elapsed = time.time() - t0
    record = {
        "image": os.path.basename(image_path),
        "orig_img_h": img_h,
        "orig_img_w": img_w,
        "persons": persons,
        "vehicles": vehicles,
        **simple_objects,
        "timing": {"total_time": round(elapsed, 3), "num_requests": total_requests},
    }
    return record


# ---------------------------------------------------------------------------
# Process one image (thread entry point)
# ---------------------------------------------------------------------------

def process_one(server, image_path, output_dir, iou_threshold, mask_threshold, timeout):
    img_stem = Path(image_path).stem
    out_path = os.path.join(output_dir, f"{img_stem}.json")
    try:
        record = annotate_image(server, image_path, iou_threshold, mask_threshold, timeout)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        return {
            "image": record["image"], "status": "success",
            "persons": len(record["persons"]),
            "vehicles": len(record["vehicles"]),
            "time": record["timing"]["total_time"],
        }
    except ImageTimeoutError as e:
        return {"image": os.path.basename(image_path), "status": "timeout", "error": str(e)}
    except Exception as e:
        return {"image": os.path.basename(image_path), "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

IS_TTY = sys.stdout.isatty()


def _log(msg):
    """Print with timestamp when output is redirected to file."""
    if IS_TTY and tqdm:
        tqdm.write(msg)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def server_worker(server, image_queue, output_dir, iou_threshold, mask_threshold,
                   timeout, pbar, stats_lock, stats):
    """Worker thread: pulls images from shared queue, sends all prompts to its dedicated server."""
    while True:
        try:
            image_path = image_queue.get_nowait()
        except Exception:
            break
        info = process_one(server, image_path, output_dir, iou_threshold, mask_threshold, timeout)
        with stats_lock:
            stats["summaries"].append(info)
            if info["status"] == "success":
                stats["success"] += 1
                stats["persons"] += info.get("persons", 0)
                stats["vehicles"] += info.get("vehicles", 0)
            elif info["status"] == "timeout":
                stats["timeouts"] += 1
                stats["skipped_images"].append(info["image"])
            else:
                stats["errors"] += 1
                _log(f"  失败: {info['image']} — {info.get('error', '')}")
            done = stats["success"] + stats["errors"] + stats["timeouts"]
            if pbar:
                pbar.update(1)
            if done % 500 == 0 and done > 0:
                elapsed = time.time() - stats["start_time"]
                avg = elapsed / done
                remaining = stats["total"] - done
                eta_min = avg * remaining / 60
                _log(f"  [stats] {done}/{stats['total']}, "
                     f"成功 {stats['success']}, 超时 {stats['timeouts']}, 失败 {stats['errors']}, "
                     f"avg {avg:.2f}s/img, ETA {eta_min:.1f} min, "
                     f"throughput {done/elapsed:.2f} img/s")


def main():
    args = parse_args()

    if args.servers:
        servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    else:
        servers = list(DEFAULT_SERVERS)

    n_prompts = len(ALL_PROMPTS)
    print(f"\n{'='*60}")
    print(f"  IPC Train 全量标注 (per-server worker)")
    print(f"  服务实例: {len(servers)} 个")
    print(f"  每张图 prompt 数: {n_prompts}")
    print(f"  架构: 每个服务独立处理不同图片")
    print(f"  IoU 阈值: {args.iou_threshold}  |  Mask 包含阈值: {args.mask_containment_threshold}")
    print(f"  单请求超时: {args.timeout}s (超时则跳过整张图片)")
    print(f"  断点续跑: {'是' if args.resume else '否'}")
    print(f"{'='*60}\n")

    pool = ServerPool(servers)
    print("[health] 检查服务状态:")
    alive = pool.check_health()
    if not alive:
        sys.exit("\n错误: 没有可用的服务实例。")
    print(f"\n[info] 可用服务: {len(alive)} 个 (即并发 worker 数)\n")

    image_dir = Path(args.image_dir)
    images = sorted(
        str(p) for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"[info] 图片总数: {len(images)}")
    print(f"[info] 总请求约 {len(images) * n_prompts}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        existing = {f.replace(".json", "")
                    for f in os.listdir(args.output_dir)
                    if f.endswith(".json") and f not in ("summary.json",)}
        before = len(images)
        images = [p for p in images if Path(p).stem not in existing]
        print(f"[resume] 已有 {before - len(images)} 张, 剩余 {len(images)} 张\n", flush=True)

    if not images:
        print("[info] 所有图片已处理完毕。")
        return

    import queue
    image_queue = queue.Queue()
    for img_path in images:
        image_queue.put(img_path)

    stats_lock = threading.Lock()
    stats = {
        "summaries": [], "success": 0, "errors": 0, "timeouts": 0,
        "persons": 0, "vehicles": 0, "skipped_images": [],
        "total": len(images), "start_time": time.time(),
    }

    pbar = tqdm(total=len(images), desc="全量标注", unit="img") if (tqdm and IS_TTY) else None

    threads = []
    for server in alive:
        t = threading.Thread(
            target=server_worker,
            args=(server, image_queue, args.output_dir,
                  args.iou_threshold, args.mask_containment_threshold,
                  args.timeout, pbar, stats_lock, stats),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if pbar:
        pbar.close()

    elapsed = time.time() - stats["start_time"]
    done = stats["success"] + stats["errors"] + stats["timeouts"]

    summary = {
        "task": "train_all_annotation",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "servers": alive,
            "num_workers": len(alive),
            "iou_threshold": args.iou_threshold,
            "mask_containment_threshold": args.mask_containment_threshold,
            "prompts_per_image": n_prompts,
            "request_timeout": args.timeout,
        },
        "total_images": len(images),
        "success": stats["success"],
        "errors": stats["errors"],
        "timeouts": stats["timeouts"],
        "total_persons": stats["persons"],
        "total_vehicles": stats["vehicles"],
        "timing": {
            "total_elapsed_seconds": round(elapsed, 2),
            "avg_per_image_seconds": round(elapsed / max(done, 1), 3),
            "throughput_images_per_second": round(done / max(elapsed, 0.01), 2),
        },
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if stats["skipped_images"]:
        skipped_path = os.path.join(args.output_dir, "skipped_timeout.txt")
        with open(skipped_path, "w", encoding="utf-8") as f:
            for img_name in stats["skipped_images"]:
                f.write(f"{img_name}\n")

    print(f"\n{'='*60}")
    print(f"  全量标注完成!")
    print(f"  图片: {len(images)}  |  成功: {stats['success']}  |  超时跳过: {stats['timeouts']}  |  失败: {stats['errors']}")
    print(f"  person: {stats['persons']}  |  vehicle: {stats['vehicles']}")
    print(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
    print(f"  吞吐: {done/max(elapsed,0.01):.2f} img/s")
    print(f"  结果: {args.output_dir}")
    if stats["skipped_images"]:
        print(f"  超时跳过列表: {os.path.join(args.output_dir, 'skipped_timeout.txt')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
