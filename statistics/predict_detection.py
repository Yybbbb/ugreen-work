#!/usr/bin/env python3
"""
Run inference on the val set (7927 images) using the RGB-only iter_22000 checkpoint.

For each image, computes per-attribute detection presence and pixel ratios,
streaming results to per_image_detection.csv.

Also tracks the worst badcase images (most FP / most FN) and saves their
prediction masks for later HTML visualization.

Usage:
  cd /data1/work/MichaelYu/segment-color/experiment/PaddleSeg
  CUDA_VISIBLE_DEVICES=0 python /data1/work/MichaelYu/segment-color/statistics/predict_detection.py
"""

import argparse
import csv
import heapq
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_STATS_ROOT    = Path(__file__).resolve().parent                         # segment-color/statistics
_SEGCOLOR_ROOT = _STATS_ROOT.parent                                      # segment-color
_PADDLESEG_ROOT = _SEGCOLOR_ROOT / "experiment" / "PaddleSeg"            # .../PaddleSeg
_POSEPRIOR_ROOT = _PADDLESEG_ROOT / "experiments" / "clothes_color_poseprior_stdc2"
_RGBONLY_ROOT   = _PADDLESEG_ROOT / "experiments" / "clothes_color_rgb_only"

for p in [_PADDLESEG_ROOT, _POSEPRIOR_ROOT, _RGBONLY_ROOT]:
    sys.path.insert(0, str(p))

import dataset_rgb          # noqa: F401 – register ClothesColorRGBDataset
import clothes_color_model  # noqa: F401 – register PPLiteSegWithColorHeads

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, utils as paddle_utils

# ── Constants ─────────────────────────────────────────────────────────────────
SEG_CLASS_NAMES = ["background", "upper", "lower", "dress",
                   "hat", "glasses", "mask", "hair"]
NUM_CLASSES = len(SEG_CLASS_NAMES)

# Colourmap for 8-class segmentation (BGR for OpenCV saving)
SEG_COLORS_BGR = {
    0: (0,   0,   0),     # background – black
    1: (0,   0,   255),   # upper      – red
    2: (0,   255, 0),     # lower      – lime
    3: (255, 0,   0),     # dress      – blue
    4: (0,   255, 255),   # hat        – yellow
    5: (255, 0,   255),   # glasses    – magenta
    6: (128, 128, 0),     # mask       – teal
    7: (255, 255, 255),   # hair       – white
}

# Note: Images are resized to 256x256 for model input, but seg_label retains
# original dimensions. Model output is interpolated back to seg_label size.
# Therefore ratios must use the actual valid pixel count per image.


# ── Badcase tracker (min-heap of worst-N images) ──────────────────────────────

class BadcaseTracker:
    """Track top-N worst images by FP count and FN count across all attributes.

    Uses min-heaps to keep the N largest scores in memory without storing
    data for all 7927 images.
    """

    def __init__(self, top_n=20):
        self.top_n = top_n
        # Min-heap entries: (score, image_rel, data_dict)
        self.fp_heap = []
        self.fn_heap = []

    @staticmethod
    def _push(heap, top_n, score, image_rel, data):
        """Push to min-heap; pop smallest if over capacity."""
        if len(heap) < top_n:
            heapq.heappush(heap, (score, image_rel, data))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, image_rel, data))

    def update(self, image_rel, row, pred_mask, gt_mask, rgb_bgr):
        """Track this image for both FP and FN heaps."""
        fp_total = sum(row[f"{n}_fp"] for n in SEG_CLASS_NAMES)
        fn_total = sum(row[f"{n}_fn"] for n in SEG_CLASS_NAMES)

        data = {"rgb_bgr": rgb_bgr,
                "pred_mask": pred_mask.copy(), "gt_mask": gt_mask.copy(),
                "row": {k: v for k, v in row.items() if k != "image_rel"}}

        self._push(self.fp_heap, self.top_n, fp_total, image_rel, data)
        self._push(self.fn_heap, self.top_n, fn_total, image_rel, data)

    def get_top_fp(self):
        return sorted(self.fp_heap, key=lambda x: x[0], reverse=True)

    def get_top_fn(self):
        return sorted(self.fn_heap, key=lambda x: x[0], reverse=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Detection analysis inference")
    p.add_argument("--config", type=str,
                   default=str(_RGBONLY_ROOT / "configs" / "optC_scratch_rgb.yml"))
    p.add_argument("--model_path", type=str,
                   default=str(_PADDLESEG_ROOT / "experiments"
                               / "pp_liteseg_stdc2_clothes_color_2head_rgb_3ch_256_optC_scratch"
                               / "checkpoints" / "iter_22000" / "model.pdparams"))
    p.add_argument("--csv_path", type=str,
                   default=str(_STATS_ROOT / "per_image_detection.csv"))
    p.add_argument("--badcase_dir", type=str,
                   default=str(_STATS_ROOT / "badcase"),
                   help="Directory to save badcase masks and meta")
    p.add_argument("--split_path", type=str, default=None,
                   help="Override val_dataset split path (e.g., train.txt)")
    p.add_argument("--dataset_mode", type=str, default="val",
                   help="Dataset mode: train or val")
    p.add_argument("--device", type=str, default="gpu")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--top_badcase", type=int, default=20,
                   help="Number of worst FP and worst FN images to save")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ──
    if args.device != "cpu":
        device = f"{args.device}:{args.device_id}"
    else:
        device = args.device
    paddle_utils.set_device(device)

    # ── Build model & dataset (patch split_path / mode if overridden) ──
    cfg = Config(args.config)
    if args.split_path is not None:
        cfg.dic["val_dataset"]["split_path"] = args.split_path
    if args.dataset_mode != "val":
        cfg.dic["val_dataset"]["mode"] = args.dataset_mode
    builder = SegBuilder(cfg)
    model = builder.model
    val_dataset = builder.val_dataset

    # ── Load checkpoint ──
    logger.info(f"Loading checkpoint: {args.model_path}")
    paddle_utils.load_entire_model(model, args.model_path)
    model.eval()

    # ── DataLoader ──
    loader = paddle.io.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        return_list=True,
    )
    total = len(loader)
    logger.info(f"Val dataset size: {len(val_dataset)}, batches: {total}")

    # ── CSV field names ──
    fieldnames = ["image_rel"]
    for n in SEG_CLASS_NAMES:
        fieldnames += [f"{n}_gt_present", f"{n}_pred_present",
                       f"{n}_gt_ratio", f"{n}_pred_ratio",
                       f"{n}_gt_pixels", f"{n}_pred_pixels",
                       f"{n}_tp", f"{n}_fp", f"{n}_fn"]
    fieldnames += ["total_valid", "mIoU", "Acc"]

    # ── Open CSV for streaming write ──
    csv_path = Path(args.csv_path)
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader()

    # ── Badcase tracker ──
    tracker = BadcaseTracker(top_n=args.top_badcase)

    # ── Inference loop ──
    t0 = time.time()
    for i, data in enumerate(loader):
        # data keys: "img" [1, 3, 256, 256], "seg_label" [1, 1, 256, 256],
        #            "upper_label", "lower_label", "image_rel"
        img = data["img"]                    # [1, 3, 256, 256]
        seg_gt_tensor = data["seg_label"]    # [1, 1, 256, 256] or [1, 256, 256]

        # Extract image_rel string
        image_rel_raw = data["image_rel"]
        if isinstance(image_rel_raw, (list, tuple)):
            image_rel = str(image_rel_raw[0])
        elif hasattr(image_rel_raw, 'item'):
            image_rel = str(image_rel_raw.item())
        else:
            image_rel = str(image_rel_raw)

        # Fix seg_gt shape
        seg_gt = seg_gt_tensor.numpy()
        if seg_gt.ndim == 4:
            seg_gt = seg_gt[0, 0]
        elif seg_gt.ndim == 3:
            seg_gt = seg_gt[0]

        # ── Forward ──
        with paddle.no_grad():
            seg_logit_list, upper_logits, lower_logits = model(img)

        seg_logit = seg_logit_list[0]  # [1, 8, 256, 256]
        if seg_logit.shape[2:] != seg_gt.shape[-2:]:
            seg_logit = F.interpolate(
                seg_logit, size=seg_gt.shape[-2:],
                mode="bilinear", align_corners=False)
        seg_pred = paddle.argmax(seg_logit, axis=1).numpy()  # [1, H, W]
        seg_pred = seg_pred[0].astype(np.uint8)

        # ── Per-class detection & ratio ──
        row = {"image_rel": image_rel}
        mask = seg_gt != 255
        total_valid = int(mask.sum())

        # Overall pixel metrics
        correct = (seg_pred == seg_gt) & mask
        acc = float(correct.sum()) / max(1, total_valid)

        iou_list = []
        for c in range(NUM_CLASSES):
            pred_c = seg_pred == c
            gt_c   = seg_gt == c

            # Pixel counts
            gt_pixels   = int((gt_c & mask).sum())
            pred_pixels = int((pred_c & mask).sum())
            inter       = int((pred_c & gt_c & mask).sum())
            union       = int(((pred_c | gt_c) & mask).sum())

            # Detection
            gt_present   = 1 if gt_pixels > 0 else 0
            pred_present = 1 if pred_pixels > 0 else 0
            tp = 1 if (gt_present == 1 and pred_present == 1) else 0
            fp = 1 if (gt_present == 0 and pred_present == 1) else 0
            fn = 1 if (gt_present == 1 and pred_present == 0) else 0

            # Ratios: use actual valid pixel count as denominator
            gt_ratio   = gt_pixels / max(1, total_valid)
            pred_ratio = pred_pixels / max(1, total_valid)

            iou_list.append(inter / max(1, union))

            name = SEG_CLASS_NAMES[c]
            row[f"{name}_gt_present"]   = gt_present
            row[f"{name}_pred_present"] = pred_present
            row[f"{name}_gt_ratio"]     = round(gt_ratio, 8)
            row[f"{name}_pred_ratio"]   = round(pred_ratio, 8)
            row[f"{name}_gt_pixels"]    = gt_pixels
            row[f"{name}_pred_pixels"]  = pred_pixels
            row[f"{name}_tp"]           = tp
            row[f"{name}_fp"]           = fp
            row[f"{name}_fn"]           = fn

        row["total_valid"] = total_valid
        row["mIoU"] = round(float(np.mean(iou_list)), 6)
        row["Acc"]  = round(acc, 6)

        # ── Write row to CSV (streaming) ──
        writer.writerow(row)

        # ── Track badcase (defer RGB loading to second pass) ──
        if (sum(row[f"{n}_fp"] for n in SEG_CLASS_NAMES) > 0 or
            sum(row[f"{n}_fn"] for n in SEG_CLASS_NAMES) > 0):
            tracker.update(image_rel, row, seg_pred, seg_gt, None)

        # ── Log progress ──
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            logger.info(f"[{i+1}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s  "
                        f"mIoU={row['mIoU']:.4f}  Acc={row['Acc']:.4f}")

        # ── Periodically clear GPU cache ──
        if (i + 1) % 1000 == 0:
            paddle.device.cuda.empty_cache()

    # ── Close CSV ──
    csv_f.close()
    elapsed = time.time() - t0
    logger.info(f"Inference finished: {total} images in {elapsed:.0f}s ({elapsed/total:.3f}s/img)")
    logger.info(f"CSV saved to: {csv_path}")

    # ── Save badcase data to disk ──
    _save_badcases(args, tracker, val_dataset)

    # ── Quick summary ──
    logger.info("=== Detection Summary (global) ===")
    _print_summary(csv_path)


def _save_badcases(args, tracker, val_dataset):
    """Save badcase masks as .npy files and metadata as .json, load RGB images."""
    badcase_dir = Path(args.badcase_dir)
    masks_dir = badcase_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(val_dataset.dataset_root)

    # Gather unique badcase images
    fp_list = tracker.get_top_fp()
    fn_list = tracker.get_top_fn()

    seen = {}
    for score, img_rel, data in fp_list:
        if img_rel not in seen:
            seen[img_rel] = data
            seen[img_rel]["badge"] = "fp"
        else:
            seen[img_rel]["badge"] = "both"

    for score, img_rel, data in fn_list:
        if img_rel not in seen:
            seen[img_rel] = data
            seen[img_rel]["badge"] = "fn"
        else:
            seen[img_rel]["badge"] = "both"

    meta = {"images": [], "top_n": args.top_badcase}

    for img_rel, data in seen.items():
        # Safe filename from image_rel
        safe_name = img_rel.replace("/", "_").replace("\\", "_")
        stem = Path(img_rel).stem

        # Save pred mask (resize to 256x256 for consistent display)
        pred_mask_resized = cv2.resize(
            data["pred_mask"].astype(np.uint8), (256, 256),
            interpolation=cv2.INTER_NEAREST)
        pred_path = masks_dir / f"{stem}_pred.npy"
        np.save(pred_path, pred_mask_resized)

        # Save gt mask (resize to 256x256)
        gt_mask_resized = cv2.resize(
            data["gt_mask"].astype(np.uint8), (256, 256),
            interpolation=cv2.INTER_NEAREST)
        gt_path = masks_dir / f"{stem}_gt.npy"
        np.save(gt_path, gt_mask_resized)

        # Load original RGB (resize to 256x256)
        img_path = dataset_root / img_rel
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_bgr = cv2.resize(img_bgr, (256, 256))
        else:
            img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)

        rgb_path = masks_dir / f"{stem}_rgb.png"
        cv2.imwrite(str(rgb_path), img_bgr)

        meta["images"].append({
            "image_rel": img_rel,
            "stem": stem,
            "badge": data["badge"],
            "row": data["row"],
            "fp_total": sum(data["row"].get(f"{n}_fp", 0) for n in SEG_CLASS_NAMES),
            "fn_total": sum(data["row"].get(f"{n}_fn", 0) for n in SEG_CLASS_NAMES),
        })

    # Sort: FP-then-FN, then by score
    meta["images"].sort(key=lambda x: (-x["fp_total"], -x["fn_total"]))

    meta_path = badcase_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Badcase data saved to: {badcase_dir}  "
                f"({len(meta['images'])} unique images)")


def _print_summary(csv_path):
    """Compute and print global detection metrics from CSV."""
    import csv as csv_mod
    counts = {}
    for n in SEG_CLASS_NAMES:
        counts[n] = {"tp": 0, "fp": 0, "fn": 0, "gt_present": 0, "pred_present": 0}

    with open(csv_path, "r") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            for n in SEG_CLASS_NAMES:
                counts[n]["tp"] += int(row[f"{n}_tp"])
                counts[n]["fp"] += int(row[f"{n}_fp"])
                counts[n]["fn"] += int(row[f"{n}_fn"])
                counts[n]["gt_present"] += int(row[f"{n}_gt_present"])
                counts[n]["pred_present"] += int(row[f"{n}_pred_present"])

    for n in SEG_CLASS_NAMES:
        c = counts[n]
        prec = c["tp"] / max(1, c["tp"] + c["fp"])
        rec  = c["tp"] / max(1, c["tp"] + c["fn"])
        f1   = 2 * prec * rec / max(1e-8, prec + rec)
        logger.info(f"  {n:12s}: Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  "
                    f"GT_present={c['gt_present']}  Pred_present={c['pred_present']}  "
                    f"TP={c['tp']}  FP={c['fp']}  FN={c['fn']}")


if __name__ == "__main__":
    main()
