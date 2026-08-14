"""FastAPI server for repeated single-region Florence-2 inference."""

import argparse
import json
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils


STATIC_DIR = Path(__file__).resolve().parent / "static"
TASK_TOKEN = "<REGION_TO_DESCRIPTION>"
DEFAULT_CHECKPOINT = (
    "/data/work/MichaelYu/florence-caption/multi_region_description/checkpoints/"
    "qwen_single_region_dhash10_grouped_gpu1_ep1_bs24_lr4e6/final"
)


def pixel_bbox_to_loc(bbox_xyxy: list[float], width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    loc = [
        math.floor(x1 / (width / 1000.0)),
        math.floor(y1 / (height / 1000.0)),
        math.floor(x2 / (width / 1000.0)),
        math.floor(y2 / (height / 1000.0)),
    ]
    return [max(0, min(999, int(value))) for value in loc]


def validate_pixel_bbox(bbox: list[float], width: int, height: int) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values")
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError("bbox values must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"bbox {bbox} is outside image bounds {width}x{height}")


def build_single_region_prompt(loc_box: list[int]) -> str:
    if len(loc_box) != 4:
        raise ValueError("loc box must contain four values")
    return TASK_TOKEN + "".join(f"<loc_{value}>" for value in loc_box)


def patch_flash_attn_import() -> None:
    original_get_imports = dynamic_module_utils.get_imports

    def patched_get_imports(filename):
        imports = original_get_imports(filename)
        if str(filename).endswith("modeling_florence2.py"):
            imports = [name for name in imports if name != "flash_attn"]
        return imports

    dynamic_module_utils.get_imports = patched_get_imports


class SingleRegionPredictor:
    def __init__(
        self,
        checkpoint: str,
        precision: str = "bf16",
        device: Optional[str] = None,
        max_new_tokens: int = 96,
        num_beams: int = 1,
    ):
        self.checkpoint = checkpoint
        self.precision = precision
        self.device_name = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.lock = threading.Lock()
        self.loaded = False
        self.processor = None
        self.model = None
        self.device = None
        self.model_dtype = None

    def load(self) -> None:
        if self.loaded:
            return
        with self.lock:
            if self.loaded:
                return
            patch_flash_attn_import()
            self.device = torch.device(
                self.device_name or ("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            self.model_dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[self.precision]
            self.processor = AutoProcessor.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
                local_files_only=True,
                use_fast=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                torch_dtype=self.model_dtype,
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="eager",
            ).to(self.device)
            self.model.eval()
            self.loaded = True

    def predict(self, image: Image.Image, bbox_xyxy: list[float]) -> dict:
        self.load()
        width, height = image.size
        validate_pixel_bbox(bbox_xyxy, width, height)
        loc_box = pixel_bbox_to_loc(bbox_xyxy, width, height)
        prompt = build_single_region_prompt(loc_box)
        started = time.perf_counter()
        with self.lock:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=640,
            )
            inputs = {
                key: value.to(
                    device=self.device,
                    dtype=self.model_dtype if torch.is_floating_point(value) else value.dtype,
                )
                for key, value in inputs.items()
            }
            autocast_enabled = self.device.type == "cuda" and self.model_dtype != torch.float32
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=self.model_dtype,
                enabled=autocast_enabled,
            ):
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    do_sample=False,
                    no_repeat_ngram_size=0,
                )
            description = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy],
            "bbox_loc_0_999": loc_box,
            "prompt": prompt,
            "description": description,
            "elapsed_ms": round(elapsed_ms, 1),
        }


predictor = SingleRegionPredictor(
    checkpoint=os.environ.get("FLORENCE_SINGLE_REGION_CHECKPOINT", DEFAULT_CHECKPOINT),
    precision=os.environ.get("FLORENCE_SINGLE_REGION_PRECISION", "bf16"),
    device=os.environ.get("FLORENCE_SINGLE_REGION_DEVICE", "cuda:0"),
    max_new_tokens=int(os.environ.get("FLORENCE_SINGLE_REGION_MAX_NEW_TOKENS", "96")),
    num_beams=int(os.environ.get("FLORENCE_SINGLE_REGION_NUM_BEAMS", "1")),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="Florence Single-Region Person Description", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(predictor.device or torch.device("cuda:0"))
    return {
        "ready": predictor.loaded,
        "task": "REGION_TO_DESCRIPTION",
        "execution_mode": "one HTTP request and one model.generate per bbox",
        "checkpoint": predictor.checkpoint,
        "precision": predictor.precision,
        "device": str(predictor.device),
        "gpu_name": gpu_name,
        "max_new_tokens": predictor.max_new_tokens,
        "num_beams": predictor.num_beams,
    }


@app.post("/api/predict-one")
async def predict_one(
    image: UploadFile = File(...),
    bbox: str = Form(...),
):
    try:
        bbox_xyxy = json.loads(bbox)
        if not isinstance(bbox_xyxy, list):
            raise ValueError("bbox must be a JSON array")
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("uploaded image is empty")
        pil_image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
        result = predictor.predict(pil_image, [float(value) for value in bbox_xyxy])
        return {
            "image_width": pil_image.width,
            "image_height": pil_image.height,
            "inference_calls": 1,
            **result,
        }
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"inference failed: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-region Florence web demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8512)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
