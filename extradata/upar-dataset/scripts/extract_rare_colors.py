#!/usr/bin/env python3
"""
Extract UPAR samples where upper OR lower clothing color is rare.
Rare colors: red, green, brown, yellow, pink, purple, other.

整合 train + val_task2 中有稀有颜色标签的数据条目。
"""
import os
from pathlib import Path

# ── paths ──
UPAR_ROOT = Path("/data1/work/MichaelYu/segment-color/extradata/upar-dataset/data/annotations/phase1")
TRAIN_CSV = UPAR_ROOT / "train/train.csv"
VAL_QUERIES_CSV = UPAR_ROOT / "val_task2/val_queries.csv"
OUTPUT = Path("/data1/work/MichaelYu/segment-color/extradata/upar-dataset/rare_color_samples.txt")

# ── UPAR 40-attribute column indices (0-based, excluding image path column) ──
# UpperBody colors: columns 8-19
UPPER_COLOR_START = 8
UPPER_COLORS = ["Black", "Blue", "Brown", "Green", "Grey", "Orange",
                "Pink", "Purple", "Red", "White", "Yellow", "Other"]

# LowerBody colors: columns 21-32
LOWER_COLOR_START = 21
LOWER_COLORS = ["Black", "Blue", "Brown", "Green", "Grey", "Orange",
                "Pink", "Purple", "Red", "White", "Yellow", "Other"]

# Rare colors (by name)
RARE_COLOR_NAMES = {"red", "green", "brown", "yellow", "pink", "purple", "other"}

# Map color name → offset within upper/lower block
RARE_UPPER_INDICES = [i for i, name in enumerate(UPPER_COLORS) if name.lower() in RARE_COLOR_NAMES]
RARE_LOWER_INDICES = [i for i, name in enumerate(LOWER_COLORS) if name.lower() in RARE_COLOR_NAMES]

print(f"Rare upper color indices (offset): {RARE_UPPER_INDICES} → {[UPPER_COLORS[i] for i in RARE_UPPER_INDICES]}")
print(f"Rare lower color indices (offset): {RARE_LOWER_INDICES} → {[LOWER_COLORS[i] for i in RARE_LOWER_INDICES]}")


def has_rare_color(values: list[int]) -> bool:
    """Check if any rare upper OR lower color flag is 1."""
    for idx in RARE_UPPER_INDICES:
        if int(values[UPPER_COLOR_START + idx]) == 1:
            return True
    for idx in RARE_LOWER_INDICES:
        if int(values[LOWER_COLOR_START + idx]) == 1:
            return True
    return False


def process_file(csv_path: Path, has_image_path: bool):
    """Yield lines (as written in output) for samples with rare colors."""
    results = []
    with open(csv_path, "r") as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if has_image_path:
                image_path = parts[0]
                values = parts[1:]
            else:
                image_path = None
                values = parts

            if len(values) < 40:
                continue

            if has_rare_color(values):
                results.append(line)

    return results


# ── main ──
train_results = process_file(TRAIN_CSV, has_image_path=True)
val_results = process_file(VAL_QUERIES_CSV, has_image_path=False)

print(f"\nTrain rare-color samples : {len(train_results)} / 97,670")
print(f"Val_task2 rare-color samples: {len(val_results)} / 3,463")

total = len(train_results) + len(val_results)

with open(OUTPUT, "w") as f:
    # Write header
    f.write("# UPAR rare-color extraction: red, green, brown, yellow, pink, purple, other\n")
    f.write(f"# Train samples: {len(train_results)}\n")
    f.write(f"# Val_task2 query samples: {len(val_results)}\n")
    f.write(f"# Total: {total}\n")
    f.write("# Format: same as original CSV (train has image_path prefix, val_task2 has attributes only)\n")
    f.write("# Age-Young,Age-Adult,Age-Old,Gender-Female,Hair-Length-Short,Hair-Length-Long,Hair-Length-Bald,UpperBody-Length-Short,UpperBody-Color-Black,UpperBody-Color-Blue,UpperBody-Color-Brown,UpperBody-Color-Green,UpperBody-Color-Grey,UpperBody-Color-Orange,UpperBody-Color-Pink,UpperBody-Color-Purple,UpperBody-Color-Red,UpperBody-Color-White,UpperBody-Color-Yellow,UpperBody-Color-Other,LowerBody-Length-Short,LowerBody-Color-Black,LowerBody-Color-Blue,LowerBody-Color-Brown,LowerBody-Color-Green,LowerBody-Color-Grey,LowerBody-Color-Orange,LowerBody-Color-Pink,LowerBody-Color-Purple,LowerBody-Color-Red,LowerBody-Color-White,LowerBody-Color-Yellow,LowerBody-Color-Other,LowerBody-Type-Trousers&Shorts,LowerBody-Type-Skirt&Dress,Accessory-Backpack,Accessory-Bag,Accessory-Glasses-Normal,Accessory-Glasses-Sun,Accessory-Hat\n")
    f.write("# --- train samples ---\n")
    for line in train_results:
        f.write(line + "\n")
    f.write("# --- val_task2 query samples ---\n")
    for line in val_results:
        f.write(line + "\n")

print(f"\n✅ Written to: {OUTPUT}")
print(f"   Total: {total} lines ({len(train_results)} train + {len(val_results)} val_task2 queries)")
