#!/usr/bin/env python3
"""Serve the RL comparison demo: index.html + bbox-overlaid person crops.

Run:
    python webdemo/rl_server.py --port 8013
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

LOGGER = logging.getLogger("webdemo.rl_server")
BUILD_DIR = Path(__file__).resolve().parent / "rl_build"
BBOX_COLOR = (255, 64, 64, 255)
BBOX_WIDTH  = 4


def draw_bbox(image_path: str, bbox: Optional[List[float]]) -> bytes:
    img = Image.open(image_path).convert("RGB")
    if bbox and len(bbox) == 4:
        draw = ImageDraw.Draw(img, "RGBA")
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0, min(x1, img.width  - 1))
        y1 = max(0, min(y1, img.height - 1))
        x2 = max(0, min(x2, img.width  - 1))
        y2 = max(0, min(y2, img.height - 1))
        draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=BBOX_WIDTH)
    if max(img.size) > 900:
        r = 900 / max(img.size)
        img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=86)
    return buf.getvalue()


def create_app() -> FastAPI:
    samples_path = BUILD_DIR / "samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"{samples_path} not found — run build_rl_demo.py first")
    samples: List[Dict[str, Any]] = json.loads(samples_path.read_text(encoding="utf-8"))
    LOGGER.info("rl_server: loaded %d sample image mappings", len(samples))
    cache: Dict[int, bytes] = {}

    app = FastAPI(title="Florence RL Comparison Demo")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        p = BUILD_DIR / "index.html"
        if not p.exists():
            raise HTTPException(500, "index.html missing — run build_rl_demo.py")
        return HTMLResponse(p.read_text(encoding="utf-8"))

    @app.get("/rl/img/{idx}")
    def image(idx: int) -> Response:
        if idx < 0 or idx >= len(samples):
            raise HTTPException(404, "index out of range")
        if idx in cache:
            return Response(content=cache[idx], media_type="image/jpeg")
        item = samples[idx]
        img_path = item.get("image", "")
        if not img_path or not Path(img_path).exists():
            raise HTTPException(404, f"source image not found: {img_path}")
        data = draw_bbox(img_path, item.get("bbox_xyxy"))
        cache[idx] = data
        return Response(content=data, media_type="image/jpeg")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "samples": len(samples)}

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(
        "webdemo.rl_server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
