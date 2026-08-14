#!/usr/bin/env python3
"""Shared, dependency-light utilities for Florence person SFT data."""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence


SUPPORTED_REGION_TASKS = {
    "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>",
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return result


def clip_bbox(
    bbox: Sequence[float], width: int, height: int
) -> List[float]:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError(f"image width must be a positive integer, got {width!r}")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError(f"image height must be a positive integer, got {height!r}")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox must contain four coordinates, got {bbox!r}")

    values = [_finite_number(value, f"bbox[{index}]") for index, value in enumerate(bbox)]
    x1 = min(max(values[0], 0.0), float(width))
    y1 = min(max(values[1], 0.0), float(height))
    x2 = min(max(values[2], 0.0), float(width))
    y2 = min(max(values[3], 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox must have positive area after clipping, got {bbox!r}")
    return [x1, y1, x2, y2]


def quantize_coordinate(value: float, dimension: int) -> int:
    coordinate = _finite_number(value, "coordinate")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError(f"dimension must be a positive integer, got {dimension!r}")
    quantized = math.floor(coordinate / (float(dimension) / 1000.0))
    return min(max(int(quantized), 0), 999)


def quantize_bbox(
    bbox: Sequence[float], width: int, height: int
) -> List[int]:
    x1, y1, x2, y2 = clip_bbox(bbox, width, height)
    return [
        quantize_coordinate(x1, width),
        quantize_coordinate(y1, height),
        quantize_coordinate(x2, width),
        quantize_coordinate(y2, height),
    ]


def validate_locations(locations: Sequence[int]) -> List[int]:
    if not isinstance(locations, (list, tuple)) or len(locations) != 4:
        raise ValueError(f"location list must contain four integers, got {locations!r}")
    result = []
    for value in locations:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999:
            raise ValueError(f"location must be an integer in [0, 999], got {value!r}")
        result.append(value)
    return result


def build_region_prompt(task: str, locations: Sequence[int]) -> str:
    if task not in SUPPORTED_REGION_TASKS:
        raise ValueError(f"Unsupported Florence region task: {task!r}")
    return task + "".join(f"<loc_{value}>" for value in validate_locations(locations))


def extract_task_text(value: Any, task: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        task_value = value.get(task)
        if isinstance(task_value, str):
            return task_value.strip()
    raise ValueError(f"Could not extract pure task text for {task}: {value!r}")


def validate_relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"relative path must be a non-empty string, got {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must stay relative to its split root: {value!r}")
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as input_file:
        document = json.load(input_file)
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return document


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield row


def _temporary_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    seen = set()
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be an object, got {type(row).__name__}")
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("Every JSONL row requires a non-empty sample_id")
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample_id: {sample_id}")
                seen.add(sample_id)
                output_file.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                count += 1
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return count


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, ensure_ascii=False, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while True:
            chunk = input_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
