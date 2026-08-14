#!/usr/bin/env python3
"""Batch extract frames from valid .dat files — 1 frame per 2 seconds."""

import subprocess
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

INPUT_BASE = Path("/nfs/public/IPC/test/HangchengPeng/目标检测遍历维度采集/dat_data")
OUTPUT_BASE = Path("/data1/work/MichaelYu/segment-color/extradata/extract_frames")
MIN_SIZE_MB = 10
WORKERS = 2  # low concurrency to avoid OOM on 4K HEVC decode
FPS = "1/2"
SCALE = "1920:-1"  # downscale to 1080p to save space and memory

DIRS = [
    "2026_3_26猫咖_目标检测",
    "2026_3_30商场_目标检测",
    "2026_4_17_小区_目标检测",
]


def collect_tasks():
    tasks = []
    for d in DIRS:
        src_dir = INPUT_BASE / d
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.glob("*.dat")):
            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb < MIN_SIZE_MB:
                continue
            out_dir = OUTPUT_BASE / f.stem
            tasks.append((f, out_dir, size_mb))
    return tasks


def extract_one(args):
    dat_path, out_dir, size_mb, idx, total = args

    # Skip if already has frames
    if out_dir.is_dir():
        existing = list(out_dir.glob("*.jpg"))
        if existing:
            return (str(dat_path), True, f"[{idx}/{total}] SKIP {dat_path.name} (already {len(existing)} frames)", len(existing))

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cmd = [
        "ffmpeg", "-nostdin",
        "-i", str(dat_path),
        "-vf", f"fps={FPS},scale={SCALE}",
        "-q:v", "2",
        "-threads", "1",
        "-loglevel", "error",
        "-y",
        str(out_dir / "%06d.jpg"),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        elapsed = time.time() - t0
        if result.returncode == 0:
            frames = len(list(out_dir.glob("*.jpg")))
            msg = f"[{idx}/{total}] OK {dat_path.name} → {frames} frames ({size_mb:.0f}MB, {elapsed:.0f}s)"
            return (str(dat_path), True, msg, frames)
        else:
            err = result.stderr[:200] if result.stderr else f"rc={result.returncode}"
            msg = f"[{idx}/{total}] FAIL {dat_path.name}: {err}"
            return (str(dat_path), False, msg, 0)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        msg = f"[{idx}/{total}] TIMEOUT {dat_path.name} ({elapsed:.0f}s)"
        return (str(dat_path), False, msg, 0)
    except Exception as e:
        elapsed = time.time() - t0
        msg = f"[{idx}/{total}] ERROR {dat_path.name}: {e}"
        return (str(dat_path), False, msg, 0)


def main():
    tasks = collect_tasks()
    total = len(tasks)
    print(f"Total valid .dat files: {total}")
    print(f"Workers: {WORKERS}")
    print(f"FPS: {FPS} (1 frame per 2s)")
    print(f"Scale: {SCALE}")
    print(f"Output: {OUTPUT_BASE}")
    print()

    args_list = [(dat, out, size, i, total) for i, (dat, out, size) in enumerate(tasks, 1)]

    completed = 0
    failed = 0
    total_frames = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(extract_one, a): a for a in args_list}

        for future in as_completed(futures):
            path, success, msg, frames = future.result()
            if success:
                completed += 1
                total_frames += frames
            else:
                failed += 1

            print(msg)

            if (completed + failed) % 10 == 0:
                elapsed = time.time() - t_start
                rate = (completed + failed) / elapsed if elapsed > 0 else 0
                eta = (total - completed - failed) / rate if rate > 0 else 0
                print(f"  Progress: {completed+failed}/{total} | OK:{completed} FAIL:{failed} | "
                      f"Frames:{total_frames} | {rate:.2f} files/s | ETA:{eta/60:.0f}min")

    total_elapsed = time.time() - t_start
    print()
    print(f"{'='*60}")
    print(f"DONE! Completed: {completed}, Failed: {failed}, Total: {total}")
    print(f"Total frames: {total_frames}")
    print(f"Elapsed: {total_elapsed/60:.1f} min")
    print(f"Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
