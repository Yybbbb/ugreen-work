import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_qwen_attribute_extraction.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_qwen_attribute_extraction", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_field_normalization_handles_common_qwen_variants(self):
        self.assertEqual(
            self.mod.normalize_value("upper_garment.length", "long-sleeved"),
            "long-sleeve",
        )
        self.assertEqual(
            self.mod.normalize_value("upper_garment.type", "long-sleeved top"),
            "top",
        )
        self.assertEqual(self.mod.normalize_value("shoes.color", "gray"), "grey")
        self.assertIsNone(self.mod.normalize_value("head.accessories", "none"))

    def test_dedup_prefers_latest_success_over_failed_retry(self):
        rows = [
            {"id": "a", "error": None, "attributes": {"gender": "male"}},
            {"id": "a", "error": "timeout", "attributes": None},
            {"id": "b", "error": "bad json", "attributes": None},
        ]
        dedup = self.mod.deduplicate_rows(rows)
        self.assertEqual(dedup["a"]["attributes"]["gender"], "male")
        self.assertEqual(dedup["b"]["error"], "bad json")

    def test_known_field_metrics_separate_unknown_extra(self):
        gt = {"age_group": "adult", "gender": "unknown"}
        pred = {"age_group": "adult", "gender": "male"}
        counts = self.mod.score_attribute_pair(gt, pred)
        self.assertEqual(counts["tp"], 1)
        self.assertEqual(counts["fp"], 0)
        self.assertEqual(counts["fn"], 0)
        self.assertEqual(counts["unknown_extra"], 1)

    def test_normalize_value_set_handles_lists(self):
        self.assertEqual(
            self.mod.normalize_value_set("upper_garment.color", "brown"),
            ["brown"],
        )
        self.assertEqual(
            self.mod.normalize_value_set("upper_garment.color", ["black", "white"]),
            ["black", "white"],
        )
        # unknown elements are dropped, duplicates removed, order preserved
        self.assertEqual(
            self.mod.normalize_value_set("shoes.color", ["gray", "gray", "unknown"]),
            ["grey"],
        )
        self.assertEqual(self.mod.normalize_value_set("shoes.color", []), [])
        self.assertEqual(self.mod.normalize_value_set("shoes.color", None), [])

    def test_list_valued_gt_scored_as_hit_not_fabrication(self):
        # Regression: list-valued GT used to collapse to None and count any
        # assertion as fabrication / unknown. It must now match an element.
        list_gt = {"upper_garment": {"color": ["black", "white"]}}
        scalar_gt = {"upper_garment": {"color": "white"}}
        pred = {"upper_garment": {"color": "white"}}
        list_counts = self.mod.score_attribute_pair(list_gt, pred)
        scalar_counts = self.mod.score_attribute_pair(scalar_gt, pred)
        # list GT matches via the 'white' element -> a true positive, not unknown
        self.assertEqual(list_counts["tp"], 1)
        self.assertEqual(list_counts["fp"], 0)
        self.assertEqual(list_counts["fn"], 0)
        self.assertEqual(list_counts["known"], 1)
        self.assertEqual(list_counts["unknown_extra"], 0)
        # and it agrees with the equivalent scalar GT
        self.assertEqual(list_counts["tp"], scalar_counts["tp"])
        self.assertAlmostEqual(list_counts["soft_tp"], scalar_counts["soft_tp"])

    def test_list_valued_gt_partial_match_uses_best_element(self):
        # pred matches one of two GT elements -> full soft_tp via max overlap
        gt = {"upper_garment": {"color": ["black", "white"]}}
        pred = {"upper_garment": {"color": "white"}}
        counts = self.mod.score_attribute_pair(gt, pred)
        self.assertAlmostEqual(counts["soft_tp"], 1.0)
        # a pred matching neither element is a soft mismatch, still not fabrication
        miss = self.mod.score_attribute_pair(gt, {"upper_garment": {"color": "red"}})
        self.assertEqual(miss["tp"], 0)
        self.assertEqual(miss["known"], 1)
        self.assertEqual(miss["unknown_extra"], 0)


if __name__ == "__main__":
    unittest.main()
