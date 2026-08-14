import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "finalize_qwen_attribute_evaluation.py"


def load_module():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("finalize_qwen_attribute_evaluation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.predictions = [
            {"sample_id": "a", "prediction": "caption a", "attributes": {"gender": "male"}},
            {"sample_id": "b", "prediction": "caption b", "attributes": {"gender": "female"}},
        ]

    def test_success_wins_over_later_failed_duplicate(self):
        rows = [
            {"id": "a", "error": None, "attributes": {"gender": "male"}},
            {"id": "a", "error": "timeout", "attributes": None},
        ]
        selected = self.mod.successful_rows_by_id(rows)
        self.assertEqual(selected["a"]["attributes"]["gender"], "male")

    def test_only_unresolved_ids_enter_retry_input(self):
        extraction = [
            {"id": "a", "error": None, "attributes": {"gender": "male"}},
            {"id": "b", "error": "bad response", "attributes": None},
        ]
        retry = self.mod.retry_input_rows(self.predictions, extraction)
        self.assertEqual([row["sample_id"] for row in retry], ["b"])

    def test_merge_is_ordered_and_requires_success_for_every_prediction(self):
        extraction = [
            {"id": "a", "error": None, "attributes": {"gender": "male"}},
            {"id": "b", "error": "bad response", "attributes": None},
        ]
        retry = [{"id": "b", "error": None, "attributes": {"gender": "female"}}]
        merged = self.mod.merge_successful_rows(self.predictions, extraction, retry)
        self.assertEqual([row["id"] for row in merged], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "missing successful Qwen extraction"):
            self.mod.merge_successful_rows(self.predictions, extraction, [])


if __name__ == "__main__":
    unittest.main()
