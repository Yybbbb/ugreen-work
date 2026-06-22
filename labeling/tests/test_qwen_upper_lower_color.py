import json
import sys
from pathlib import Path

import pytest


LABELING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABELING_DIR))

from qwen_upper_lower_color import (  # noqa: E402
    build_prompt,
    call_qwen,
    extract_garment_regions,
    parse_json_from_text,
    process_entry,
    xywh_norm_to_qwen_xyxy,
)


def test_xywh_norm_to_qwen_xyxy_converts_to_0_999_grid():
    assert xywh_norm_to_qwen_xyxy([0.1, 0.2, 0.3, 0.4]) == [100, 200, 400, 599]


def test_xywh_norm_to_qwen_xyxy_clamps_out_of_range_values():
    assert xywh_norm_to_qwen_xyxy([-0.1, 0.9, 1.2, 0.3]) == [0, 899, 999, 999]


def test_extract_garment_regions_skips_unknown_and_missing_bbox(tmp_path):
    annotation_path = tmp_path / "sample.json"
    annotation = {
        "sample_id": "sample/001",
        "image": {"path": "images/sample/001.jpg", "width": 100, "height": 200},
        "attributes": {
            "upper": {"label": "upper", "bbox": [0.1, 0.2, 0.3, 0.4], "score": 0.95},
            "lower": {"label": "unknown", "bbox": None, "score": 0.0},
        },
    }
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

    regions = extract_garment_regions(annotation_path)

    assert regions == [
        {
            "part": "upper",
            "source_label": "upper",
            "bbox_xywh_norm": [0.1, 0.2, 0.3, 0.4],
            "bbox_qwen_xyxy_0_999": [100, 200, 400, 599],
            "annotation_score": 0.95,
        }
    ]


def test_extract_garment_regions_derives_bbox_from_lip_mask(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    from PIL import Image

    mask_path = mask_dir / "sample.png"
    mask = Image.new("L", (10, 20), 0)
    pixels = mask.load()
    for y in range(4, 10):
        for x in range(2, 7):
            pixels[x, y] = 5
    mask.save(mask_path)

    annotation_path = tmp_path / "sample.json"
    annotation = {
        "sample_id": "sample/002",
        "image": {"path": "images/sample.jpg", "width": 10, "height": 20},
        "attributes": {
            "upper": {
                "label": "upper",
                "bbox": None,
                "score": None,
                "mask": {
                    "format": "lip_palette_ids",
                    "path": str(mask_path),
                    "lip_ids": [5],
                },
            },
            "lower": {"label": "unknown", "bbox": None},
        },
    }
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

    regions = extract_garment_regions(annotation_path)

    assert regions == [
        {
            "part": "upper",
            "source_label": "upper",
            "bbox_xywh_norm": [0.2, 0.2, 0.5, 0.3],
            "bbox_qwen_xyxy_0_999": [200, 200, 699, 500],
            "annotation_score": 0.0,
            "bbox_source": "mask",
        }
    ]


def test_parse_json_from_text_returns_none_for_malformed_embedded_json():
    assert parse_json_from_text("prefix {not valid json} suffix") is None


def test_call_qwen_disables_thinking(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image bytes")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    import qwen_upper_lower_color

    monkeypatch.setattr(qwen_upper_lower_color.requests, "post", fake_post)

    call_qwen("http://example.test/v1", "model", "prompt", image_path, 12, 0.0, 123)

    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_process_entry_writes_error_json_when_image_missing(tmp_path):
    output_dir = tmp_path / "out"
    annotation_path = tmp_path / "sample.json"
    missing_image_path = tmp_path / "missing.jpg"
    annotation_path.write_text(
        json.dumps(
            {
                "image": {"path": "images/sample.jpg", "width": 10, "height": 20},
                "attributes": {
                    "upper": {"label": "upper", "bbox": [0.1, 0.2, 0.3, 0.4]},
                    "lower": {"label": "lower", "bbox": [0.2, 0.5, 0.4, 0.3]},
                },
            }
        ),
        encoding="utf-8",
    )

    ok = process_entry(
        image_path=missing_image_path,
        annotation_path=annotation_path,
        prompt_template="{regions_json}\n{color_schema_json}",
        output_path=output_dir / "sample.json",
        base_url="http://example.test/v1",
        model="model",
        timeout=1,
        temperature=0.0,
        max_tokens=10,
        dry_run=False,
    )

    saved = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
    assert ok is False
    assert saved["status"] == "error"
    assert saved["error_type"] == "FileNotFoundError"
    assert "Image not found" in saved["error"]


def test_build_prompt_includes_regions_and_color_schema():
    template = (
        "Regions:\n{regions_json}\n"
        "Colors:\n{color_schema_json}\n"
        "Return JSON only."
    )
    regions = [
        {
            "part": "upper",
            "bbox_qwen_xyxy_0_999": [100, 200, 400, 599],
        }
    ]

    prompt = build_prompt(template, regions)

    assert '"part": "upper"' in prompt
    assert '"bbox_qwen_xyxy_0_999": [' in prompt
    assert '"english": "black"' in prompt
    assert '"chinese": "黑色"' in prompt
