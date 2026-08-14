import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "person_sft_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("person_sft_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BboxTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_quantize_bbox_uses_floor_and_clamp(self):
        self.assertEqual(
            self.mod.quantize_bbox([-4.0, 50.9, 640.9, 900.0], 1280, 720),
            [0, 70, 500, 999],
        )

    def test_quantize_bbox_rejects_non_positive_area(self):
        with self.assertRaisesRegex(ValueError, "positive area"):
            self.mod.quantize_bbox([10, 10, 10, 20], 100, 100)

    def test_build_prompt_uses_native_task_and_four_locations(self):
        self.assertEqual(
            self.mod.build_region_prompt(
                "<REGION_TO_CATEGORY>", [1, 2, 3, 4]
            ),
            "<REGION_TO_CATEGORY><loc_1><loc_2><loc_3><loc_4>",
        )

    def test_build_prompt_rejects_invalid_location(self):
        with self.assertRaisesRegex(ValueError, "location"):
            self.mod.build_region_prompt(
                "<REGION_TO_DESCRIPTION>", [1, 2, 3, 1000]
            )

    def test_extract_task_text_unwraps_processor_mapping(self):
        self.assertEqual(
            self.mod.extract_task_text(
                {"<REGION_TO_DESCRIPTION>": "A person in a blue jacket."},
                "<REGION_TO_DESCRIPTION>",
            ),
            "A person in a blue jacket.",
        )

    def test_extract_task_text_accepts_direct_text(self):
        self.assertEqual(
            self.mod.extract_task_text("A person.", "<REGION_TO_CATEGORY>"),
            "A person.",
        )

    def test_extract_task_text_rejects_wrong_task_mapping(self):
        with self.assertRaisesRegex(ValueError, "task text"):
            self.mod.extract_task_text(
                {"<CAPTION>": "A room."}, "<REGION_TO_CATEGORY>"
            )


class JsonlTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_atomic_writer_round_trips_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "rows.jsonl"
            rows = [
                {"sample_id": "a", "label": "first"},
                {"sample_id": "b", "label": "second"},
            ]

            count = self.mod.write_jsonl_atomic(path, rows)

            self.assertEqual(count, 2)
            self.assertEqual(list(self.mod.iter_jsonl(path)), rows)
            self.assertFalse(path.with_name("rows.jsonl.tmp").exists())

    def test_atomic_writer_rejects_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = [
                {"sample_id": "duplicate", "label": "first"},
                {"sample_id": "duplicate", "label": "second"},
            ]

            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                self.mod.write_jsonl_atomic(path, rows)

            self.assertFalse(path.exists())
            self.assertFalse(path.with_name("rows.jsonl.tmp").exists())

    def test_sha256_file_matches_written_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.txt"
            path.write_bytes(b"abc")
            self.assertEqual(
                self.mod.sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
