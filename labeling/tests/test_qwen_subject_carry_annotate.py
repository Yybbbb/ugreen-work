from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "object_detection_0309-0429"
    / "qwen_subject_carry_annotate.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("qwen_subject_carry_annotate", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_subject_label_path_preserves_image_subdirectories(tmp_path):
    mod = load_module()
    data_root = tmp_path / "data"
    output_root = data_root / "object_detection_0309-0429" / "subject-label"

    output_path, split_path = mod.subject_label_paths(
        "object_detection_0309-0429/images/2026_3_09_park/example_p00.jpg",
        data_root,
        output_root,
    )

    assert output_path == output_root / "2026_3_09_park" / "example_p00.json"
    assert split_path == "object_detection_0309-0429/subject-label/2026_3_09_park/example_p00.json"


def test_parse_and_validate_accepts_only_three_allowed_labels():
    mod = load_module()
    text = """
    ```json
    {"hand_carry_object": "with", "back_carry_object": "unknow"}
    ```
    """

    parsed = mod.validate_parsed(mod.parse_json_from_text(text))

    assert parsed == {
        "hand_carry_object": "with",
        "back_carry_object": "unknow",
    }


def test_validate_rejects_unknown_spelling_and_missing_keys():
    mod = load_module()

    assert mod.validate_parsed(
        {"hand_carry_object": "unknown", "back_carry_object": "none"}
    ) is None
    assert mod.validate_parsed({"hand_carry_object": "with"}) is None


def test_prompt_includes_non_exhaustive_examples_and_exclusions():
    mod = load_module()
    prompt = mod.build_prompt()

    assert "Examples are not exhaustive" in prompt
    assert "shopping bag" in prompt
    assert "stroller handle" in prompt
    assert "hiking pack" in prompt
    assert "Clothing, hats, masks, glasses, shoes, hair, scarves" in prompt
    assert "Background objects do not count" in prompt
    assert "Objects carried by other people do not count" in prompt


def test_split_file_third_column_is_added_or_replaced(tmp_path):
    mod = load_module()
    data_root = tmp_path / "data"
    split_file = data_root / "object_detection_0309-0429_random200_color_label.txt"
    split_file.parent.mkdir(parents=True)
    split_file.write_text(
        "\n".join(
            [
                "object_detection_0309-0429/images/a/img1.jpg object_detection_0309-0429/color-label/a/img1.json",
                "object_detection_0309-0429/images/b/img2.jpg object_detection_0309-0429/color-label/b/img2.json old/subject.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entries = mod.load_split_entries(split_file, data_root, data_root / "object_detection_0309-0429" / "subject-label")
    mod.write_split_with_subject_column(split_file, entries)

    assert split_file.read_text(encoding="utf-8").splitlines() == [
        "object_detection_0309-0429/images/a/img1.jpg object_detection_0309-0429/color-label/a/img1.json object_detection_0309-0429/subject-label/a/img1.json",
        "object_detection_0309-0429/images/b/img2.jpg object_detection_0309-0429/color-label/b/img2.json object_detection_0309-0429/subject-label/b/img2.json",
    ]


def test_process_entry_dry_run_writes_subject_json(tmp_path):
    mod = load_module()
    data_root = tmp_path / "data"
    image_path = data_root / "object_detection_0309-0429" / "images" / "a" / "img1.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-a-real-image")
    output_path = data_root / "object_detection_0309-0429" / "subject-label" / "a" / "img1.json"

    entry = mod.SplitEntry(
        image_rel="object_detection_0309-0429/images/a/img1.jpg",
        color_label_rel="object_detection_0309-0429/color-label/a/img1.json",
        subject_label_rel="object_detection_0309-0429/subject-label/a/img1.json",
        image_path=image_path,
        output_path=output_path,
    )

    result = mod.process_entry(
        entry=entry,
        prompt=mod.build_prompt(),
        base_url="http://example.test/v1",
        model="test-model",
        timeout=1,
        temperature=0.0,
        max_tokens=128,
        dry_run=True,
    )

    assert result["status"] == "success"
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["parsed"] == {
        "hand_carry_object": "unknow",
        "back_carry_object": "unknow",
    }
