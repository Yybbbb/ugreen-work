import json
import sys
from pathlib import Path


LABELING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABELING_DIR))

from export_review_samples import (  # noqa: E402
    collect_review_candidates,
    export_review_samples,
)


def test_collect_review_candidates_skips_non_success_and_missing_parsed(tmp_path):
    annotation_dir = tmp_path / "annotation"
    image_dir = tmp_path / "images"
    annotation_dir.mkdir()
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"a")

    valid = annotation_dir / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "status": "success",
                "image_path": str(image_dir / "a.jpg"),
                "parsed": {"upper": [{"label": "white", "confidence": 0.9}], "lower": []},
            }
        ),
        encoding="utf-8",
    )
    (annotation_dir / "error.json").write_text(
        json.dumps({"status": "error", "image_path": "x.jpg", "parsed": {}}),
        encoding="utf-8",
    )
    (annotation_dir / "missing_parsed.json").write_text(
        json.dumps({"status": "success", "image_path": "y.jpg"}),
        encoding="utf-8",
    )

    candidates = collect_review_candidates(annotation_dir)

    assert len(candidates) == 1
    assert candidates[0]["image_path"] == image_dir / "a.jpg"
    assert candidates[0]["upper"] == [{"label": "white", "confidence": 0.9}]
    assert candidates[0]["lower"] == []


def test_export_review_samples_copies_images_and_writes_json(tmp_path):
    annotation_dir = tmp_path / "annotation"
    source_image_dir = tmp_path / "source_images"
    output_image_dir = tmp_path / "images"
    annotation_dir.mkdir()
    source_image_dir.mkdir()

    image_a = source_image_dir / "look_a.jpg"
    image_b = source_image_dir / "look_b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    (annotation_dir / "a.json").write_text(
        json.dumps(
            {
                "status": "success",
                "image_path": str(image_a),
                "parsed": {
                    "upper": [{"label": "white", "confidence": 0.9}],
                    "lower": [{"label": "black", "confidence": 0.8}],
                },
            }
        ),
        encoding="utf-8",
    )
    (annotation_dir / "b.json").write_text(
        json.dumps(
            {
                "status": "success",
                "image_path": str(image_b),
                "parsed": {
                    "upper": [{"label": "blue", "confidence": 0.7}],
                    "lower": [{"label": "gray", "confidence": 0.6, "inferred": True}],
                },
            }
        ),
        encoding="utf-8",
    )

    json_path = tmp_path / "review.json"
    rows = export_review_samples(
        annotation_dir=annotation_dir,
        output_image_dir=output_image_dir,
        json_path=json_path,
        sample_size=1,
        seed=7,
    )

    assert len(rows) == 1
    row = rows[0]
    copied_path = output_image_dir / row["image_name"]
    assert copied_path.exists()
    assert copied_path.parent == output_image_dir
    assert row["image_name"] == copied_path.name
    assert "upper" in row
    assert "lower" in row

    saved_rows = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(saved_rows) == 1
    assert saved_rows[0] == row
    assert list(saved_rows[0]) == ["image_name", "upper", "lower"]
