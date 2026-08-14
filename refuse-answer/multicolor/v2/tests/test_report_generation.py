import json
import tempfile
from pathlib import Path

from segment_color_import import add_repo_path

add_repo_path()

from generate_html_report import generate_report


def test_generate_report_contains_metrics_confusion_and_images():
    jpeg_bytes = b"fake-jpeg-content"
    report = {
        "sets": {
            "pure": {"total": 4, "processed": 4, "errors": 0},
            "multi": {"total": 4, "processed": 4, "errors": 0},
        },
        "confusion_matrix": {
            "actual_pure_pred_pure_tp": 0,
            "actual_pure_pred_multi_fn": 4,
            "actual_multi_pred_pure_fp": 4,
            "actual_multi_pred_multi_tn": 0,
        },
        "metrics_processed_only": {
            "accuracy": 0.0,
            "pure_recall": 0.0,
            "multi_recall": 0.0,
            "pure_precision": 0.0,
            "multi_precision": 0.0,
        },
        "results": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i in range(4):
            pure_path = tmp_path / f"pure_{i}.jpg"
            multi_path = tmp_path / f"multi_{i}.jpg"
            pure_path.write_bytes(jpeg_bytes)
            multi_path.write_bytes(jpeg_bytes)
            report["results"].append(_badcase_item(pure_path, "pure", "multi", "second_ratio>=0.10"))
            report["results"].append(_badcase_item(multi_path, "multi", "pure", "pure"))

        input_path = Path(tmp) / "eval.json"
        output_path = Path(tmp) / "report.html"
        input_path.write_text(json.dumps(report), encoding="utf-8")

        generate_report(input_path, output_path)

        html = output_path.read_text(encoding="utf-8")
        assert "Accuracy" in html
        assert "Confusion Matrix" in html
        assert "Actual Multi / Pred Pure" in html
        assert 'src="data:image/jpeg;base64,' in html
        assert 'src="data:image/svg+xml;base64,' in html
        assert "Upper Mask" in html
        assert '<img src="' + str(tmp_path / "multi_0.jpg") not in html
        assert html.count("<img src=") == 8


def _badcase_item(path, gt_label, prediction, reason):
    return {
        "id": path.name,
        "path": str(path),
        "gt_label": gt_label,
        "prediction": prediction,
        "detection": {
            "decision_reason": reason,
            "main_ratio": 0.7,
            "second_ratio": 0.2,
            "minor_total_ratio": 0.3,
            "n_effective_colors": 3,
            "n_pixels": 100,
        },
    }
