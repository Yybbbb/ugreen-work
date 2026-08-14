#!/usr/bin/env python3
"""
Memory-efficient inference on train.txt using the RGB-only PPLiteSegWithColorHeads model.

Key features:
  - No pre-scan: image dimensions collected via fast binary header reads on-the-fly
  - Uses DataLoader with batch_size=1 for memory efficiency
  - Periodic cache clearing and result flushing
  - Supports resume from checkpoint

Output:
  - results.jsonl: per-image metrics including original W, H, area, etc.
  - pred_upper.txt / pred_lower.txt: color predictions
  - global_metrics.json: aggregate metrics

Usage:
  cd /data1/work/MichaelYu/segment-color/experiment/PaddleSeg
  python /data1/work/MichaelYu/segment-color/refuse-answer/smallcrop/infer_train_rgb.py
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR    = Path(__file__).resolve().parent
_PADDLESEG_ROOT = Path("/data1/work/MichaelYu/segment-color/experiment/PaddleSeg")
_RGB_EXP        = _PADDLESEG_ROOT / "experiments" / "clothes_color_rgb_only"
_POSEPRIOR_EXP  = _PADDLESEG_ROOT / "experiments" / "clothes_color_poseprior_stdc2"

for _p in [_PADDLESEG_ROOT, _RGB_EXP, _POSEPRIOR_EXP]:
    sys.path.insert(0, str(_p))

import dataset_rgb              # noqa: F401 – registers ClothesColorRGBDataset
import clothes_color_model      # noqa: F401 – registers PPLiteSegWithColorHeads
from dataset_rgb import ClothesColorRGBDataset

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, utils as paddle_utils

# ── Constants ────────────────────────────────────────────────────────────────
SEG_CLASS_NAMES = ["background", "upper", "lower", "dress",
                    "hat", "glasses", "mask", "hair"]

COLOR_NAMES = ["unknown", "black", "white", "gray", "red", "yellow",
               "green", "blue", "purple", "pink", "orange", "brown"]

CONFIG_PATH     = _RGB_EXP / "configs" / "optC_scratch_rgb.yml"
CHECKPOINT_PATH = _PADDLESEG_ROOT / "experiments" / "pp_liteseg_stdc2_clothes_color_2head_rgb_3ch_256_optC_scratch" / "checkpoints" / "iter_22000" / "model.pdparams"


def parse_args():
    p = argparse.ArgumentParser(description="Infer train.txt with RGB model")
    p.add_argument("--config", type=str, default=str(CONFIG_PATH))
    p.add_argument("--model_path", type=str, default=str(CHECKPOINT_PATH))
    p.add_argument("--save_dir", type=str, default=str(_SCRIPT_DIR / "infer_train_rgb"))
    p.add_argument("--device", type=str, default="gpu")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def fast_image_size(filepath):
    """
    Read image (W, H) from binary header without full decode.
    Supports JPEG and PNG. Returns (0, 0) on failure.
    Reads at most 32KB (worst case for PNG).
    """
    try:
        with open(filepath, 'rb') as f:
            head = f.read(32)
            if len(head) < 2:
                return (0, 0)

            # JPEG: look for SOF0/SOF2 marker (0xFF 0xC0 or 0xFF 0xC2)
            if head[:2] == b'\xff\xd8':
                f.seek(2)
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    i = 0
                    while i < len(chunk) - 1:
                        if chunk[i] == 0xFF and chunk[i+1] in (0xC0, 0xC1, 0xC2):
                            if i + 8 < len(chunk):
                                h = struct.unpack('>HH', chunk[i+5:i+9])
                                return (h[1], h[0])  # (W, H)
                            else:
                                return (0, 0)
                        i += 1
                    # Don't search forever
                    if f.tell() > 32768:
                        break
                return (0, 0)

            # PNG: IHDR at offset 16
            if head[:8] == b'\x89PNG\r\n\x1a\n':
                w, h = struct.unpack('>II', head[16:24])
                return (w, h)

            # GIF
            if head[:3] in (b'GIF', b'GIF'):
                w, h = struct.unpack('<HH', head[6:10])
                return (w, h)

            # BMP
            if head[:2] == b'BM':
                w, h = struct.unpack('<II', head[18:26])
                return (abs(w), abs(h))

            # WebP: RIFF...WEBP
            if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
                # VP8 chunk
                chunk = head[12:]
                if chunk[:4] == b'VP8 ':
                    w = struct.unpack('<H', chunk[6:8])[0] & 0x3fff
                    h = struct.unpack('<H', chunk[8:10])[0] & 0x3fff
                    return (w, h)
                elif chunk[:4] == b'VP8L':
                    bits = struct.unpack('<I', chunk[4:8])[0]
                    w = (bits & 0x3fff) + 1
                    h = ((bits >> 14) & 0x3fff) + 1
                    return (w, h)
                return (0, 0)

            return (0, 0)
    except Exception:
        return (0, 0)


def compute_seg_metrics(pred_hw, gt_hw, num_classes=8, ignore_index=255):
    mask = gt_hw != ignore_index
    correct = (pred_hw == gt_hw) & mask
    acc = float(correct.sum()) / max(1, int(mask.sum()))

    iou_per_cls = []
    prec_per_cls = []
    rec_per_cls = []
    for c in range(num_classes):
        pred_c = pred_hw == c
        gt_c = gt_hw == c
        inter = int((pred_c & gt_c & mask).sum())
        pred_n = int((pred_c & mask).sum())
        gt_n = int((gt_c & mask).sum())
        union = pred_n + gt_n - inter
        iou_per_cls.append(inter / max(1, union))
        prec_per_cls.append(inter / max(1, pred_n))
        rec_per_cls.append(inter / max(1, gt_n))

    return {
        "mIoU": float(np.mean(iou_per_cls)), "Acc": acc,
        "per_class_IoU": iou_per_cls,
        "per_class_Precision": prec_per_cls,
        "per_class_Recall": rec_per_cls,
    }


def main():
    args = parse_args()

    # ── Device ──
    if args.device != "cpu":
        device = f"{args.device}:{args.device_id}"
    else:
        device = args.device
    paddle_utils.set_device(device)
    paddle_utils.show_env_info()

    # ── Build model & dataset ──
    cfg = Config(args.config)
    builder = SegBuilder(cfg)
    model = builder.model

    train_cfg = cfg.train_dataset_cfg
    dataset_root = Path(train_cfg["dataset_root"])
    train_dataset = ClothesColorRGBDataset(
        mode="val",
        dataset_root=str(dataset_root),
        split_path=train_cfg["split_path"],
        transforms=builder.val_transforms,
        color_mode=train_cfg.get("color_mode", "ce"),
        label_cache_dir=train_cfg.get("label_cache_dir"),
    )

    # ── Load checkpoint ──
    logger.info(f"Loading checkpoint: {args.model_path}")
    paddle_utils.load_entire_model(model, args.model_path)
    model.eval()
    logger.info("Checkpoint loaded; model in eval mode.")

    # ── Output directories ──
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume support ──
    processed = set()
    results_path = save_dir / "results.jsonl"
    if args.resume and results_path.exists():
        with open(results_path, "r") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    processed.add(r["image_rel"])
                except json.JSONDecodeError:
                    continue
        logger.info(f"Resume: {len(processed)} images already processed.")

    # ── Filter ──
    all_samples = train_dataset.samples
    logger.info(f"Dataset loaded: {len(all_samples)} total samples")
    if args.limit > 0:
        all_samples = all_samples[:args.limit]
    remaining = [s for s in all_samples if s["image_rel"] not in processed]
    logger.info(f"Total: {len(all_samples)}, processed: {len(processed)}, remaining: {len(remaining)}")

    if not remaining:
        logger.info("All images already processed. Exiting.")
        return

    # ── DataLoader ──
    train_dataset.samples = remaining
    loader = paddle.io.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, return_list=True,
    )
    total = len(loader)
    logger.info(f"Inference batches: {total}")

    # ── Open output files ──
    results_f = open(results_path, "a")
    upper_f = open(save_dir / "pred_upper.txt", "a" if args.resume else "w")
    lower_f = open(save_dir / "pred_lower.txt", "a" if args.resume else "w")
    if not args.resume:
        upper_f.write("image_rel\tpred_id\tpred_name\tgt_id\tgt_name\n")
        lower_f.write("image_rel\tpred_id\tpred_name\tgt_id\tgt_name\n")

    # ── Accumulators ──
    per_class_intersect = np.zeros(8, dtype=np.int64)
    per_class_union = np.zeros(8, dtype=np.int64)
    upper_correct = 0; lower_correct = 0
    upper_total = 0; lower_total = 0
    total_done = 0

    t0 = time.time()
    for i, data in enumerate(loader):
        img = data["img"]
        seg_gt = data["seg_label"].numpy()
        upper_gt = int(data["upper_label"].numpy().item())
        lower_gt = int(data["lower_label"].numpy().item())

        # Handle image_rel
        img_rel_raw = data["image_rel"]
        if isinstance(img_rel_raw, (list, tuple)):
            image_rel = str(img_rel_raw[0])
        elif hasattr(img_rel_raw, 'item'):
            image_rel = str(img_rel_raw.item())
        else:
            image_rel = str(img_rel_raw)

        # Fix seg_gt shape
        if seg_gt.ndim == 4:
            seg_gt = seg_gt[0, 0]
        elif seg_gt.ndim == 3:
            seg_gt = seg_gt[0]

        # ── Get original image dimensions (fast binary read) ──
        img_path = dataset_root / image_rel
        orig_w, orig_h = fast_image_size(str(img_path))

        # ── Forward ──
        with paddle.no_grad():
            seg_logit_list, upper_logits, lower_logits = model(img)

        seg_logit = seg_logit_list[0]
        if seg_logit.shape[2:] != seg_gt.shape[-2:]:
            seg_logit = F.interpolate(
                seg_logit, size=seg_gt.shape[-2:],
                mode="bilinear", align_corners=False)
        seg_pred = paddle.argmax(seg_logit, axis=1).numpy()
        seg_pred = seg_pred[0].astype(np.uint8)

        upper_pred = int(paddle.argmax(upper_logits, axis=1).numpy().item())
        lower_pred = int(paddle.argmax(lower_logits, axis=1).numpy().item())

        # ── Per-image metrics ──
        m = compute_seg_metrics(seg_pred, seg_gt, num_classes=8, ignore_index=255)

        # Accumulate global seg
        for c in range(8):
            pred_c = seg_pred == c
            gt_c = seg_gt == c
            mask = seg_gt != 255
            per_class_intersect[c] += int((pred_c & gt_c & mask).sum())
            per_class_union[c] += int(((pred_c | gt_c) & mask).sum())

        # Accumulate color
        u_correct = 0; l_correct = 0
        if upper_gt != 0:
            upper_total += 1
            if upper_pred == upper_gt:
                upper_correct += 1
                u_correct = 1
        if lower_gt != 0:
            lower_total += 1
            if lower_pred == lower_gt:
                lower_correct += 1
                l_correct = 1

        # Write predictions
        upper_f.write(f"{image_rel}\t{upper_pred}\t{COLOR_NAMES[upper_pred]}\t{upper_gt}\t{COLOR_NAMES[upper_gt]}\n")
        lower_f.write(f"{image_rel}\t{lower_pred}\t{COLOR_NAMES[lower_pred]}\t{lower_gt}\t{COLOR_NAMES[lower_gt]}\n")

        # Size metrics
        short_edge = min(orig_w, orig_h)
        long_edge = max(orig_w, orig_h)
        area_kpx = (orig_w * orig_h) / 1000.0 if orig_w > 0 and orig_h > 0 else 0.0
        aspect = orig_w / orig_h if orig_h > 0 else 0.0

        # Write result
        result = {
            "image_rel": image_rel,
            "orig_W": orig_w, "orig_H": orig_h,
            "area_kpx": round(area_kpx, 2),
            "short_edge": short_edge, "long_edge": long_edge,
            "aspect_ratio": round(aspect, 4),
            "seg_mIoU": round(m["mIoU"], 6),
            "seg_Acc": round(m["Acc"], 6),
            "per_class_IoU": [round(v, 6) for v in m["per_class_IoU"]],
            "upper_gt": upper_gt, "upper_pred": upper_pred, "upper_correct": u_correct,
            "lower_gt": lower_gt, "lower_pred": lower_pred, "lower_correct": l_correct,
        }
        results_f.write(json.dumps(result) + "\n")
        total_done += 1

        # Progress
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            logger.info(
                f"[{i+1}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s  "
                f"rate={rate:.1f} img/s  mIoU={m['mIoU']:.4f}  "
                f"up: {COLOR_NAMES[upper_pred]}  lo: {COLOR_NAMES[lower_pred]}"
            )
            results_f.flush(); upper_f.flush(); lower_f.flush()

        # Memory cleanup
        del img, seg_logit_list, upper_logits, lower_logits, seg_logit, seg_pred
        if (i + 1) % 200 == 0:
            paddle.device.cuda.empty_cache()

    # ── Done ──
    results_f.close(); upper_f.close(); lower_f.close()
    elapsed = time.time() - t0
    logger.info(f"Inference finished: {total_done} images in {elapsed:.0f}s ({elapsed/total_done:.3f}s/img)")

    # ── Global metrics ──
    overall_iou = []
    for c in range(8):
        iou_c = per_class_intersect[c] / max(1, per_class_union[c])
        overall_iou.append(iou_c)
    overall_miou = float(np.mean(overall_iou))
    upper_acc = upper_correct / max(1, upper_total)
    lower_acc = lower_correct / max(1, lower_total)

    logger.info(f"=== Global Metrics (iter_22000 on train.txt) ===")
    logger.info(f"  mIoU: {overall_miou:.4f}")
    for i, n in enumerate(SEG_CLASS_NAMES):
        logger.info(f"  {n:12s}: IoU={overall_iou[i]:.4f}")
    logger.info(f"  Upper color acc: {upper_acc:.4f} ({upper_correct}/{upper_total})")
    logger.info(f"  Lower color acc: {lower_acc:.4f} ({lower_correct}/{lower_total})")

    with open(save_dir / "global_metrics.json", "w") as f:
        json.dump({
            "checkpoint": str(CHECKPOINT_PATH),
            "num_images": total_done,
            "mIoU": overall_miou,
            "class_IoU": {n: overall_iou[i] for i, n in enumerate(SEG_CLASS_NAMES)},
            "upper_color_acc": upper_acc, "lower_color_acc": lower_acc,
            "upper_correct": int(upper_correct), "upper_total": int(upper_total),
            "lower_correct": int(lower_correct), "lower_total": int(lower_total),
        }, f, indent=2)

    paddle.device.cuda.empty_cache()
    logger.info("Done.")


if __name__ == "__main__":
    main()
