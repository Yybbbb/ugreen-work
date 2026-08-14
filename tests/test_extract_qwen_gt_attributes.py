import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "extract_qwen_gt_attributes.py"


def load_module():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("extract_qwen_gt_attributes_tested", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QwenGtExtractionTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def args(self, input_path, output_path, fresh=True):
        return argparse.Namespace(
            input=input_path, output=output_path, base_url="http://qwen/v1",
            model="model", batch_size=2, workers=1, timeout=1.0,
            retries=0, max_tokens=32, fresh=fresh,
        )

    def fake_request(self, batch, **kwargs):
        return [
            {"id": str(row["sample_id"]), "attributes": {"gender": "adult"}, "raw": "{}", "error": None}
            for row in batch
        ]

    def test_blank_caption_is_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "test.jsonl"
            output = root / "attrs.jsonl"
            self.write_rows(source, [
                {"sample_id": "keep", "label": "a person"},
                {"sample_id": "drop", "label": "  \t"},
            ])
            with mock.patch.object(self.mod, "request_batch_gt", side_effect=self.fake_request):
                manifest = self.mod.run(self.args(source, output))
            rows = list(self.mod.iter_rows(output))
        self.assertEqual([row["sample_id"] for row in rows], ["keep"])
        self.assertEqual(manifest["input_samples"], 2)
        self.assertEqual(manifest["empty_caption_samples"], 1)
        self.assertEqual(manifest["samples"], 1)

    def test_fresh_mode_replaces_stale_output_in_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "test.jsonl"
            output = root / "attrs.jsonl"
            self.write_rows(source, [
                {"sample_id": "b", "label": "caption b"},
                {"sample_id": "a", "label": "caption a"},
            ])
            self.write_rows(output, [{"sample_id": "stale", "qwen_attributes": {}}])
            with mock.patch.object(self.mod, "request_batch_gt", side_effect=self.fake_request):
                manifest = self.mod.run(self.args(source, output, fresh=True))
            rows = list(self.mod.iter_rows(output))
            written_manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([row["sample_id"] for row in rows], ["b", "a"])
        self.assertEqual(manifest["resumed_from"], 0)
        self.assertEqual(written_manifest["samples"], 2)
        self.assertEqual(written_manifest["errors"], 0)

    def test_failed_batch_rows_are_retried_individually(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "test.jsonl"
            output = root / "attrs.jsonl"
            self.write_rows(source, [
                {"sample_id": "a", "label": "caption a"},
                {"sample_id": "b", "label": "caption b"},
            ])
            calls = []

            def fail_batch(batch, **kwargs):
                calls.append([row["sample_id"] for row in batch])
                if len(batch) > 1:
                    return [
                        {"id": row["sample_id"], "attributes": None, "raw": "", "error": "response ids mismatch"}
                        for row in batch
                    ]
                return self.fake_request(batch, **kwargs)

            with mock.patch.object(self.mod, "request_batch_gt", side_effect=fail_batch):
                manifest = self.mod.run(self.args(source, output))
            rows = list(self.mod.iter_rows(output))

        self.assertEqual(calls, [["a", "b"], ["a"], ["b"]])
        self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])
        self.assertEqual(manifest["individual_retries"], 2)
        self.assertEqual(manifest["errors"], 0)


if __name__ == "__main__":
    unittest.main()
