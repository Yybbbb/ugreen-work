import subprocess
from pathlib import Path


def load_rgb_image(image_path):
    image_path = Path(image_path)
    width, height = probe_image_size(image_path)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(image_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    expected = width * height * 3
    if len(raw) != expected:
        raise ValueError(f"decoded byte count mismatch for {image_path}: {len(raw)} != {expected}")

    rows = []
    pos = 0
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append([raw[pos], raw[pos + 1], raw[pos + 2]])
            pos += 3
        rows.append(row)
    return rows, width, height


def probe_image_size(image_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(image_path),
    ]
    output = subprocess.check_output(cmd, text=True).strip()
    width_s, height_s = output.split(",")
    return int(width_s), int(height_s)
