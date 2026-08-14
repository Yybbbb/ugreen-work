#!/usr/bin/env python3
"""Serve the V4A attribute demo: index.html + bbox-overlaid person crops.

Images are drawn on demand: the original full frame is returned with the
person detection bbox overlaid in red, so the viewer sees the image itself
plus which region the caption describes. Rendered bytes are cached in memory.

Run:
    python webdemo/server.py --host 0.0.0.0 --port 8012
"""

from __future__ import annotations

import argparse
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from PIL import Image, ImageDraw

LOGGER = logging.getLogger("webdemo.server")
LOGGER.setLevel(logging.INFO)

BUILD_DIR = Path(__file__).resolve().parent / "build"
BBOX_COLOR = (255, 64, 64, 255)
BBOX_WIDTH = 4


def load_samples() -> List[Dict[str, Any]]:
    path = BUILD_DIR / "samples.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python webdemo/build_demo.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def draw_bbox_overlay(image_path: str, bbox: Optional[List[float]]) -> bytes:
    img = Image.open(image_path).convert("RGB")
    if bbox and len(bbox) == 4:
        draw = ImageDraw.Draw(img, "RGBA")
        x1, y1, x2, y2 = (float(v) for v in bbox)
        # clamp to image bounds
        x1 = max(0, min(x1, img.width - 1))
        y1 = max(0, min(y1, img.height - 1))
        x2 = max(0, min(x2, img.width - 1))
        y2 = max(0, min(y2, img.height - 1))
        draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=BBOX_WIDTH)
    # cap longest side to keep payloads reasonable
    max_side = 900
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=86)
    return buf.getvalue()


def create_app() -> FastAPI:
    samples = load_samples()
    LOGGER.info("loaded %d sample image mappings", len(samples))
    cache: Dict[int, bytes] = {}

    app = FastAPI(title="Florence V4A Attribute Demo")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        path = BUILD_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=500, detail="index.html missing — run build_demo.py")
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.get("/img/{idx}")
    def image(idx: int) -> Response:
        if idx < 0 or idx >= len(samples):
            raise HTTPException(status_code=404, detail="image index out of range")
        if idx in cache:
            return Response(content=cache[idx], media_type="image/jpeg")
        item = samples[idx]
        img_path = item.get("image", "")
        if not img_path or not Path(img_path).exists():
            raise HTTPException(status_code=404, detail=f"source image not found: {img_path}")
        data = draw_bbox_overlay(img_path, item.get("bbox_xyxy"))
        cache[idx] = data
        return Response(content=data, media_type="image/jpeg")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "samples": len(samples)}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(
        "webdemo.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
