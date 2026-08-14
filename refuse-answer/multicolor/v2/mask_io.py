import ast
import json
import struct
from pathlib import Path


DEFAULT_DATA_ROOT = Path("data/LIP_clothes_accessory_unified")
DEFAULT_ANNOTATIONS_ROOT = DEFAULT_DATA_ROOT / "annotations"


def load_upper_mask_for_image(image_path, annotations_root=DEFAULT_ANNOTATIONS_ROOT, data_root=DEFAULT_DATA_ROOT):
    image_path = Path(image_path)
    annotations_root = Path(annotations_root)
    data_root = Path(data_root)
    key = searchable_key(image_path)
    annotation_path = find_annotation(key, annotations_root)
    if annotation_path is None:
        raise FileNotFoundError(f"annotation not found for key {key}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    upper = annotation.get("attributes", {}).get("upper", {})
    mask_info = upper.get("mask") if upper.get("has_mask") else None
    if not isinstance(mask_info, dict):
        raise ValueError(f"upper mask missing in {annotation_path}")
    fmt = mask_info.get("format")
    label_path = None
    lip_ids = []

    if fmt == "lip_palette_ids":
        lip_ids = sorted(int(value) for value in mask_info.get("lip_ids", []))
        if not lip_ids:
            raise ValueError(f"upper lip_ids missing in {annotation_path}")

        label_path = find_label_cache(key, data_root)
        if label_path is None:
            raise FileNotFoundError(f"label cache not found for key {key}")

        labels, width, height = load_uint8_npy(label_path)
        lip_id_set = set(lip_ids)
        mask = _mask_from_label_ids(labels, lip_id_set)
        if _mask_pixel_count(mask) == 0:
            # The local cache stores unified labels for some LIP samples
            # rather than original LIP palette ids. In that cache, 1=upper.
            mask = _mask_from_label_ids(labels, {1})
        if _mask_pixel_count(mask) == 0:
            raise ValueError(f"upper mask is empty after applying ids {lip_ids} and unified id 1")
    elif fmt == "coco_rle":
        mask, width, height = decode_coco_rle(mask_info.get("rle"))
    elif fmt == "coco_rle_list":
        mask, width, height = decode_coco_rle_list(mask_info.get("annotations"))
    else:
        raise ValueError(f"unsupported upper mask format: {fmt}")

    return {
        "mask": mask,
        "width": width,
        "height": height,
        "annotation_path": str(annotation_path),
        "label_path": str(label_path) if label_path else None,
        "lip_ids": lip_ids,
        "search_key": key,
    }


def _mask_from_label_ids(labels, label_ids):
    mask = []
    for row in labels:
        mask.append([1 if value in label_ids else 0 for value in row])
    return mask


def _mask_pixel_count(mask):
    return sum(sum(row) for row in mask)


def searchable_key(image_path):
    stem = Path(image_path).stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else stem


def find_annotation(key, annotations_root):
    root = Path(annotations_root)
    candidates = [
        root / "ipc_seg_annotation" / f"{key}.json",
        root / "lip" / "Train" / f"{key}.json",
        root / "lip" / "Val" / f"{key}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for subdir in [root / "upper_lower_person_export", root / "ipc_seg_annotation"]:
        if subdir.exists():
            matches = sorted(subdir.glob(f"*{key}*.json"))
            if matches:
                return matches[0]

    return None


def find_label_cache(key, data_root):
    root = Path(data_root)
    search_dirs = [
        root / "cache" / "clothes_accessory_poseprior" / "labels" / "annotations" / "lip" / "Train",
        root / "cache" / "clothes_accessory_poseprior" / "labels" / "annotations" / "lip" / "Val",
        root / "cache" / "clothes_accessory_poseprior" / "labels_8cls_hair" / "annotations" / "lip" / "Train",
        root / "cache" / "clothes_accessory_poseprior" / "labels_8cls_hair" / "annotations" / "lip" / "Val",
    ]
    matches = []
    for search_dir in search_dirs:
        if search_dir.exists():
            matches.extend(sorted(search_dir.glob(f"{key}__*_uint8.npy")))
    if not matches:
        return None
    matches.sort(key=lambda path: ("/labels/" not in path.as_posix(), len(path.as_posix())))
    return matches[0]


def load_uint8_npy(path):
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"not a npy file: {path}")
        major, _minor = struct.unpack("BB", f.read(2))
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        elif major == 2:
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"unsupported npy version {major} in {path}")
        header = f.read(header_len).decode("latin1").strip()
        meta = ast.literal_eval(header)
        if meta.get("descr") != "|u1":
            raise ValueError(f"expected uint8 npy, got {meta.get('descr')} in {path}")
        if meta.get("fortran_order"):
            raise ValueError(f"fortran-order npy is unsupported: {path}")
        shape = meta.get("shape")
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError(f"expected 2D npy shape, got {shape} in {path}")
        height, width = int(shape[0]), int(shape[1])
        raw = f.read(width * height)

    if len(raw) != width * height:
        raise ValueError(f"npy payload size mismatch in {path}")

    rows = []
    pos = 0
    for _ in range(height):
        row = list(raw[pos : pos + width])
        rows.append(row)
        pos += width
    return rows, width, height


def decode_coco_rle(rle):
    if not isinstance(rle, dict):
        raise ValueError("invalid coco_rle payload")
    size = rle.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError(f"invalid coco_rle size: {size}")
    height, width = int(size[0]), int(size[1])
    counts = rle.get("counts")
    if isinstance(counts, str):
        counts = _decode_compressed_counts(counts)
    elif isinstance(counts, list):
        counts = [int(value) for value in counts]
    else:
        raise ValueError("invalid coco_rle counts")

    flat = [0] * (height * width)
    pos = 0
    value = 0
    for run_len in counts:
        end = min(pos + int(run_len), len(flat))
        if value:
            for idx in range(pos, end):
                flat[idx] = 1
        pos = end
        value = 1 - value
        if pos >= len(flat):
            break

    rows = [[0 for _ in range(width)] for _ in range(height)]
    for x in range(width):
        for y in range(height):
            rows[y][x] = flat[x * height + y]
    return rows, width, height


def decode_coco_rle_list(annotations):
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("invalid coco_rle_list annotations")

    merged = None
    width = height = None
    for annotation in annotations:
        segmentation = annotation.get("segmentation") if isinstance(annotation, dict) else None
        rows, item_width, item_height = decode_coco_rle(segmentation)
        if merged is None:
            width, height = item_width, item_height
            merged = [[0 for _ in range(width)] for _ in range(height)]
        elif item_width != width or item_height != height:
            raise ValueError("coco_rle_list contains inconsistent mask sizes")

        for y in range(height):
            merged_row = merged[y]
            row = rows[y]
            for x in range(width):
                if row[x]:
                    merged_row[x] = 1

    return merged, width, height


def _decode_compressed_counts(text):
    counts = []
    idx = 0
    while idx < len(text):
        shift = 0
        value = 0
        while True:
            c = ord(text[idx]) - 48
            idx += 1
            value |= (c & 0x1F) << shift
            more = c & 0x20
            shift += 5
            if not more:
                if c & 0x10:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts
