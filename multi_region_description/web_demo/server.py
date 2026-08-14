"""
FastAPI demo server for the <REGIONS_TO_DESCRIPTIONS> multi-region checkpoint.

Supports two output modes selectable per-request:
  plan_a  (default) — descriptions separated by <sep>, positional alignment
  plan_b            — description-first with loc tokens appended, regex parsing

One POST /api/predict call:
  - Accepts an image + N pixel bboxes + optional mode
  - Builds a single <REGIONS_TO_DESCRIPTIONS> prompt
  - Runs ONE model.generate (one image encode)
  - Parses output according to the selected mode
  - Returns raw output + aligned results list
"""

import argparse
import json
import math
import os
import re
import threading
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils


STATIC_DIR = Path(__file__).resolve().parent / "static"
TASK_TOKEN = "<REGIONS_TO_DESCRIPTIONS>"
SEP = "<sep>"

DEFAULT_CHECKPOINT_PLAN_A = (
    "/data/work/MichaelYu/florence-caption/multi_region_description/"
    "checkpoints/multi_region_no_name_ep2_lr1e5/final"
)
DEFAULT_CHECKPOINT_PLAN_B = (
    "/data/work/MichaelYu/florence-caption/multi_region_description/"
    "checkpoints/multi_region_plan_b_ep2_lr1e5/final"
)

# Plan B parsing: text segment (non-greedy) + exactly 4 loc tokens (spaces allowed between them)
_PLAN_B_REGION_RE = re.compile(r"(.*?)((?:<loc_\d+>\s*){4})", re.DOTALL)
_LOC_VALUE_RE     = re.compile(r"<loc_(\d+)>")
_STRIP_TOKENS_RE  = re.compile(r"<s>|</s>|<pad>")


# --------------------------------------------------------------------------- #
# bbox helpers
# --------------------------------------------------------------------------- #

def pixel_bbox_to_loc(bbox_xyxy: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    size_per_bin_w = width / 1000.0
    size_per_bin_h = height / 1000.0
    loc = [
        math.floor(x1 / size_per_bin_w),
        math.floor(y1 / size_per_bin_h),
        math.floor(x2 / size_per_bin_w),
        math.floor(y2 / size_per_bin_h),
    ]
    return [max(0, min(999, int(v))) for v in loc]


def validate_pixel_bbox(bbox: list[float], width: int, height: int) -> None:
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(v) for v in bbox):
        raise ValueError("bbox values must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"bbox {bbox} is outside image bounds {width}x{height}")


def build_multi_region_prompt(loc_boxes: list[list[int]]) -> str:
    groups = ["".join(f"<loc_{v}>" for v in box) for box in loc_boxes]
    return f"{TASK_TOKEN}{SEP.join(groups)}"


# --------------------------------------------------------------------------- #
# model wrapper
# --------------------------------------------------------------------------- #

def _patch_flash_attn():
    orig = dynamic_module_utils.get_imports
    def patched(fn):
        imps = orig(fn)
        if str(fn).endswith("modeling_florence2.py"):
            imps = [n for n in imps if n != "flash_attn"]
        return imps
    dynamic_module_utils.get_imports = patched


class MultiRegionPredictor:
    def __init__(self, checkpoint: str, precision: str = "fp16", device: Optional[str] = None):
        self.checkpoint = checkpoint
        self.precision = precision
        self.device_name = device
        self._lock = threading.Lock()
        self._loaded = False
        self.processor = None
        self.model = None
        self.device = None
        self.model_dtype = None
        self.autocast_dtype = None

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            _patch_flash_attn()
            self.device = torch.device(
                self.device_name if self.device_name
                else ("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            self.model_dtype = dtype_map.get(self.precision, torch.float16)
            self.autocast_dtype = self.model_dtype if self.precision in ("fp16", "bf16") else None

            self.processor = AutoProcessor.from_pretrained(
                self.checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, torch_dtype=self.model_dtype,
                trust_remote_code=True, local_files_only=True,
            ).to(self.device)
            self.model.eval()
            self._loaded = True

    def predict(
        self,
        image: Image.Image,
        loc_boxes: list[list[int]],
        max_new_tokens: int = 256,
        num_beams: int = 1,
        mode: str = "plan_a",
    ) -> tuple[str, list[str]]:
        """Run one generate call; return (raw_output_text, descriptions_list).

        mode='plan_a': use processor.post_process_generation (<sep>-split)
        mode='plan_b': decode only new tokens, apply regex (desc + 4 loc-run)
        """
        self.load()
        prompt = build_multi_region_prompt(loc_boxes)
        inputs = self.processor(
            text=prompt, images=image,
            return_tensors="pt", padding=True, truncation=True, max_length=2048,
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model_dtype)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype is not None and self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                no_repeat_ngram_size=0,
            )

        if mode == "plan_b":
            # Florence-2 is encoder-decoder, so generate() returns decoder output
            # only. Keep decoder-only compatibility for other model families.
            if self.model.config.is_encoder_decoder:
                new_ids = generated_ids
            else:
                new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            raw = self.processor.batch_decode(new_ids, skip_special_tokens=False)[0]
            descriptions = _parse_plan_b(raw)
        else:
            # Plan A: keep special tokens so <sep> survives post-processor
            raw = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            result = self.processor.post_process_generation(
                raw, task=TASK_TOKEN, image_size=image.size
            )
            descriptions = result[TASK_TOKEN]["descriptions"]
        return raw, descriptions



# --------------------------------------------------------------------------- #
# Plan B output parser
# --------------------------------------------------------------------------- #

def _parse_plan_b(text: str) -> list[str]:
    """Extract descriptions from a Plan B raw output string.

    Plan B format: desc1<loc_x><loc_y><loc_x><loc_y>desc2<loc_...>...
    Returns a list of description strings (stripped).
    """
    text = _STRIP_TOKENS_RE.sub("", text).strip()
    if text.startswith(TASK_TOKEN):
        text = text[len(TASK_TOKEN):]
    return [m.group(1).strip() for m in _PLAN_B_REGION_RE.finditer(text)]


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

def create_app() -> FastAPI:
    ckpt_a   = os.environ.get("FLORENCE_CHECKPOINT_PLAN_A", DEFAULT_CHECKPOINT_PLAN_A)
    ckpt_b   = os.environ.get("FLORENCE_CHECKPOINT_PLAN_B", DEFAULT_CHECKPOINT_PLAN_B)
    precision = os.environ.get("FLORENCE_MULTI_REGION_PRECISION", "fp16")
    device    = os.environ.get("FLORENCE_MULTI_REGION_DEVICE")
    default_max_new_tokens = int(os.environ.get("FLORENCE_MULTI_REGION_MAX_NEW_TOKENS", "256"))

    # Two predictors, each loaded lazily on first use
    predictors = {
        "plan_a": MultiRegionPredictor(checkpoint=ckpt_a, precision=precision, device=device),
        "plan_b": MultiRegionPredictor(checkpoint=ckpt_b, precision=precision, device=device),
    }

    app = FastAPI(title="Florence2 Multi-Region Description Demo")

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "models": {
                "plan_a": {"checkpoint": ckpt_a, "loaded": predictors["plan_a"]._loaded},
                "plan_b": {"checkpoint": ckpt_b, "loaded": predictors["plan_b"]._loaded},
            },
            "precision": precision,
            "device": device or "auto",
            "default_max_new_tokens": default_max_new_tokens,
        }

    @app.post("/api/predict")
    async def predict(
        image: UploadFile = File(...),
        boxes_json: str = Form(...),
        mode: str = Form("plan_a"),
        max_new_tokens: Optional[int] = Form(None),
        num_beams: Optional[int] = Form(None),
    ):
        if mode not in predictors:
            raise HTTPException(status_code=400, detail=f"unknown mode '{mode}'; choose plan_a or plan_b")

        try:
            payload = json.loads(boxes_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid boxes_json: {exc}") from exc

        try:
            resolved_max_new_tokens = max_new_tokens if max_new_tokens is not None else default_max_new_tokens
            resolved_num_beams = num_beams if num_beams is not None else 1
            if resolved_max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be positive")

            image_bytes = await image.read()
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            width, height = pil_image.size

            if not isinstance(payload, list) or not payload:
                raise ValueError("boxes_json must be a non-empty JSON list")

            bboxes_pixel: list[list[float]] = []
            loc_boxes: list[list[int]] = []
            for i, item in enumerate(payload):
                if not isinstance(item, dict):
                    raise ValueError(f"box {i}: expected object")
                values = item.get("bbox")
                if not isinstance(values, list) or len(values) != 4:
                    raise ValueError(f"box {i}: bbox must contain 4 values")
                bbox = [float(v) for v in values]
                validate_pixel_bbox(bbox, width, height)
                bboxes_pixel.append(bbox)
                loc_boxes.append(pixel_bbox_to_loc(bbox, width, height))

            # -------- single inference call --------
            raw, descriptions = predictors[mode].predict(
                pil_image, loc_boxes,
                max_new_tokens=resolved_max_new_tokens,
                num_beams=resolved_num_beams,
                mode=mode,
            )

            count_match = len(descriptions) == len(bboxes_pixel)
            results: list[dict[str, Any]] = []
            for i, (bbox, loc) in enumerate(zip(bboxes_pixel, loc_boxes)):
                results.append({
                    "index": i,
                    "bbox_pixel": [round(v, 1) for v in bbox],
                    "bbox_loc_0_999": loc,
                    "description": descriptions[i] if i < len(descriptions) else None,
                })

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "image": {"filename": image.filename, "width": width, "height": height},
            "mode": mode,
            "count": len(results),
            "count_match": count_match,
            "raw": raw,
            "descriptions": descriptions,
            "results": results,
        }

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7863")))
    args = parser.parse_args()
    import uvicorn
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
