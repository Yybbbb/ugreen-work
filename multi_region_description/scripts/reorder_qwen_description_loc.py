#!/usr/bin/env python3
"""Create jointly reordered prompt/label variants of description+loc data."""

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


TASK_TOKEN = "<REGIONS_TO_DESCRIPTIONS>"
SEP_TOKEN = "<sep>"
LOC_PATTERN = re.compile(r"<loc_(-?\d+)>")
LABEL_GROUP_PATTERN = re.compile(
    r"(.*?)(<loc_-?\d+><loc_-?\d+><loc_-?\d+><loc_-?\d+>)",
    re.DOTALL,
)
DEFAULT_INPUT_DIR = Path(
    "/data/work/MichaelYu/florence-caption/multi_region_description/data/"
    "qwen-person-2to5-description-loc"
)
OUTPUT_DIR_SUFFIXES = {
    "random": "order-random",
    "area": "order-area-desc",
    "center": "order-center-left-to-right",
    "top": "order-center-top-to-bottom",
}


@dataclass(frozen=True)
class CropGroup:
    description: str
    loc_text: str
    bbox: tuple[int, int, int, int]
    original_index: int

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def doubled_center_x(self) -> int:
        x1, _, x2, _ = self.bbox
        return x1 + x2

    @property
    def doubled_center_y(self) -> int:
        _, y1, _, y2 = self.bbox
        return y1 + y2


def parse_loc_text(loc_text: str, context: str) -> tuple[int, int, int, int]:
    coordinates = tuple(int(value) for value in LOC_PATTERN.findall(loc_text))
    if len(coordinates) != 4 or loc_text != "".join(
        f"<loc_{coordinate}>" for coordinate in coordinates
    ):
        raise ValueError(f"{context}: expected exactly four contiguous loc tokens")
    return coordinates


def parse_sample(sample: dict, context: str) -> list[CropGroup]:
    prompt = sample["prompt"]
    if not prompt.startswith(TASK_TOKEN):
        raise ValueError(f"{context}: prompt does not start with {TASK_TOKEN}")

    prompt_loc_texts = prompt[len(TASK_TOKEN) :].split(SEP_TOKEN)
    label_matches = list(LABEL_GROUP_PATTERN.finditer(sample["label"]))
    if "".join(match.group(0) for match in label_matches) != sample["label"]:
        raise ValueError(f"{context}: label is not a sequence of description+loc groups")
    if len(prompt_loc_texts) != len(label_matches):
        raise ValueError(
            f"{context}: prompt has {len(prompt_loc_texts)} crops but label has "
            f"{len(label_matches)} groups"
        )

    groups = []
    for index, (prompt_loc_text, label_match) in enumerate(
        zip(prompt_loc_texts, label_matches)
    ):
        label_loc_text = label_match.group(2)
        if prompt_loc_text != label_loc_text:
            raise ValueError(f"{context}: crop {index} prompt/label loc tokens differ")
        groups.append(
            CropGroup(
                description=label_match.group(1),
                loc_text=label_loc_text,
                bbox=parse_loc_text(label_loc_text, f"{context}, crop {index}"),
                original_index=index,
            )
        )
    return groups


def reordered_groups(
    groups: list[CropGroup], mode: str, rng: random.Random
) -> list[CropGroup]:
    if mode == "random":
        reordered = groups.copy()
        rng.shuffle(reordered)
        if len(reordered) > 1 and reordered == groups:
            reordered = reordered[1:] + reordered[:1]
        return reordered
    if mode == "area":
        return sorted(
            groups,
            key=lambda group: (
                -group.area,
                group.doubled_center_x,
                group.original_index,
            ),
        )
    if mode == "center":
        return sorted(
            groups,
            key=lambda group: (
                group.doubled_center_x,
                -group.area,
                group.original_index,
            ),
        )
    if mode == "top":
        return sorted(
            groups,
            key=lambda group: (
                group.doubled_center_y,
                -group.area,
                group.original_index,
            ),
        )
    raise ValueError(f"unsupported mode: {mode}")


def build_sample(sample: dict, groups: list[CropGroup]) -> dict:
    return {
        **sample,
        "prompt": TASK_TOKEN + SEP_TOKEN.join(group.loc_text for group in groups),
        "label": "".join(group.description + group.loc_text for group in groups),
    }


def process_split(
    input_path: Path, output_paths: dict[str, Path], seed: int
) -> tuple[int, dict[str, int]]:
    rng = random.Random(seed)
    changed_counts = {mode: 0 for mode in output_paths}
    row_count = 0

    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_files = {
        mode: output_path.open("w", encoding="utf-8")
        for mode, output_path in output_paths.items()
    }
    try:
        with input_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                groups = parse_sample(sample, f"{input_path}:{line_number}")
                original_indices = [group.original_index for group in groups]
                for mode, output_file in output_files.items():
                    ordered = reordered_groups(groups, mode, rng)
                    if [group.original_index for group in ordered] != original_indices:
                        changed_counts[mode] += 1
                    output_file.write(
                        json.dumps(build_sample(sample, ordered), ensure_ascii=False) + "\n"
                    )
                row_count += 1
    finally:
        for output_file in output_files.values():
            output_file.close()

    return row_count, changed_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create jointly reordered prompt/label crop-order datasets"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_INPUT_DIR.parent,
        help="Parent directory for the output dataset directories",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(OUTPUT_DIR_SUFFIXES),
        default=list(OUTPUT_DIR_SUFFIXES),
        help="Crop-order variants to generate",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for split_index, split in enumerate(("train", "test")):
        input_path = args.input_dir / f"{split}.jsonl"
        output_paths = {
            mode: (
                args.output_root
                / f"{args.input_dir.name}-{OUTPUT_DIR_SUFFIXES[mode]}"
                / f"{split}.jsonl"
            )
            for mode in args.modes
        }
        row_count, changed_counts = process_split(
            input_path, output_paths, seed=args.seed + split_index
        )
        print(
            f"{split}: rows={row_count}, "
            + ", ".join(
                f"{mode}_changed={changed_counts[mode]}" for mode in args.modes
            )
        )


if __name__ == "__main__":
    main()
