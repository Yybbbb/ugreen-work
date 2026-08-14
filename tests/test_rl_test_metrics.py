import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "rl_test_metrics.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("rl_test_metrics", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(sample_id, caption, gt, pred, session="scene/session-a"):
    return {
        "sample_id": sample_id,
        "session": session,
        "prediction": caption,
        "ground_truth": gt,
        "attributes": pred,
    }


class EightMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_exact_scalar_match_produces_perfect_f1(self):
        rows = [row("a", "An adult male in blue clothing stands calmly here today.",
                    {"gender": "male"}, {"gender": "male"})]
        result = self.mod.score_rows(rows, self.mod.lexical_similarity)
        self.assertEqual(result["metrics"]["micro"], {"precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertEqual(result["metrics"]["macro_field_f1"], 1.0)
        self.assertEqual(result["metrics"]["fabrication_ratio"], 0.0)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def candidate(self, name, predictions):
        rows = []
        for index, (gt, pred) in enumerate(predictions):
            rows.append(row(
                str(index),
                "An adult person with dark hair wears a blue jacket and black pants with white shoes today outside.",
                {"gender": gt},
                {"gender": pred} if pred is not None else {},
                session=f"session-{index // 2}",
            ))
        return self.mod.build_candidate(name, f"/checkpoints/{name}", rows, self.mod.lexical_similarity)

    def test_bootstrap_is_deterministic_and_paired(self):
        baseline = self.candidate("v4b_sft", [("male", "male"), ("male", None), ("female", "female"), ("female", None)])
        improved = self.candidate("qwen_rl", [("male", "male"), ("male", "male"), ("female", "female"), ("female", "female")])
        first = self.mod.compare_candidates([baseline, improved], bootstrap_replicates=100, seed=17)
        second = self.mod.compare_candidates([baseline, improved], bootstrap_replicates=100, seed=17)
        self.assertEqual(first["comparisons"], second["comparisons"])
        ci = first["comparisons"]["qwen_rl"]["delta_ci95"]["joint_attribute_f1"]
        self.assertGreater(ci[0], 0.0)

    def test_bootstrap_uses_preaggregated_sessions_not_sample_reaggregation(self):
        baseline = self.candidate("v4b_sft", [("male", "male"), ("male", None)])
        improved = self.candidate("qwen_rl", [("male", "male"), ("male", "male")])
        original = self.mod._headline

        def absolute_only(candidate, sample_indices=None):
            if sample_indices is not None:
                raise AssertionError("bootstrap reaggregated raw samples")
            return original(candidate)

        with mock.patch.object(self.mod, "_headline", side_effect=absolute_only):
            self.mod.compare_candidates([baseline, improved], bootstrap_replicates=10, seed=5)

    def test_clear_eligible_improvement_is_recommended(self):
        baseline = self.candidate("v4b_sft", [("male", "male"), ("male", None), ("female", "female"), ("female", None)])
        improved = self.candidate("qwen_rl", [("male", "male"), ("male", "male"), ("female", "female"), ("female", "female")])
        result = self.mod.compare_candidates([baseline, improved], bootstrap_replicates=100, seed=7)
        self.assertEqual(result["recommendation"]["candidate"], "qwen_rl")
        self.assertTrue(result["comparisons"]["qwen_rl"]["eligible"])
        self.assertEqual(result["comparisons"]["qwen_rl"]["win_tie_loss"]["lexical"], {"win": 2, "tie": 2, "loss": 0})

    def test_f1_regression_fails_gate(self):
        baseline = self.candidate("v4b_sft", [("male", "male")] * 4)
        regressed = self.candidate("lexical_rl", [("male", "male"), ("male", "male"), ("male", None), ("male", None)])
        result = self.mod.compare_candidates([baseline, regressed], bootstrap_replicates=50, seed=1)
        comparison = result["comparisons"]["lexical_rl"]
        self.assertFalse(comparison["eligible"])
        self.assertIn("lexical_micro_f1", comparison["failed_gates"])
        self.assertEqual(result["recommendation"]["candidate"], "v4b_sft")

    def test_tiny_or_uncertain_gain_retains_v4b(self):
        baseline = self.candidate("v4b_sft", [("male", "male")] * 100 + [("male", None)])
        tiny = self.candidate("qwen_rl", [("male", "male")] * 101)
        result = self.mod.compare_candidates([baseline, tiny], bootstrap_replicates=100, seed=4)
        self.assertLessEqual(
            result["comparisons"]["qwen_rl"]["delta"]["joint_attribute_f1"], 0.005
        )
        self.assertEqual(result["recommendation"]["candidate"], "v4b_sft")

    def test_markdown_has_three_checkpoints_and_eight_metric_rows(self):
        candidates = [self.candidate(name, [("male", "male")]) for name in ("v4b_sft", "qwen_rl", "lexical_rl")]
        comparison = self.mod.compare_candidates(candidates, bootstrap_replicates=10, seed=3)
        report = self.mod.render_markdown(comparison)
        for name in ("v4b_sft", "qwen_rl", "lexical_rl"):
            self.assertIn(name, report)
        for label in self.mod.HEADLINE_METRIC_LABELS:
            self.assertIn(label, report)
        self.assertIn("Delta vs V4B", report)
        self.assertIn("95% CI", report)
        self.assertIn("Win / tie / loss", report)
        self.assertIn("Largest field changes", report)

    def test_field_diagnostics_include_soft_count_deltas(self):
        baseline = self.candidate("v4b_sft", [("male", None), ("male", "male")])
        improved = self.candidate("qwen_rl", [("male", "male"), ("male", "male")])
        result = self.mod.compare_candidates([baseline, improved], bootstrap_replicates=10, seed=2)
        gender = next(
            item for item in result["comparisons"]["qwen_rl"]["field_deltas"]
            if item["field"] == "gender"
        )
        self.assertEqual(gender["soft_tp_delta"], 1.0)
        self.assertEqual(gender["soft_fn_delta"], -1.0)
        self.assertEqual(gender["soft_fp_delta"], 0.0)

    def test_comparison_rejects_session_identity_mismatch(self):
        baseline = self.candidate("v4b_sft", [("male", "male")])
        candidate = self.candidate("qwen_rl", [("male", "male")])
        candidate["lexical"]["samples"][0]["session"] = "different"
        with self.assertRaisesRegex(ValueError, "session mismatch"):
            self.mod.compare_candidates([baseline, candidate], bootstrap_replicates=10)

    def test_soft_counts_distinguish_recall_and_fabrication(self):
        rows = [row(
            "a",
            "An adult male wears blue clothing and carries a bright green bag outdoors today.",
            {"gender": "male", "upper_garment": {"color": "blue"}},
            {"gender": "male", "upper_garment": {"color": "red"}, "shoes": {"color": "green"}},
        )]
        result = self.mod.score_rows(rows, self.mod.lexical_similarity)
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["micro"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["micro"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["micro"]["f1"], 0.5)
        self.assertAlmostEqual(metrics["fabrication_ratio"], 1.0 / 3.0)

    def test_extra_matching_and_unmatched_generation(self):
        rows = [row(
            "a",
            "An adult person wearing a watch and ring is visible in this image.",
            {"extra": ["silver watch"]},
            {"extra": ["silver watch", "gold ring"]},
        )]
        result = self.mod.score_rows(rows, self.mod.lexical_similarity)
        self.assertEqual(result["metrics"]["micro"]["recall"], 1.0)
        self.assertEqual(result["metrics"]["fabrication_ratio"], 0.5)

    def test_semantic_similarity_is_injected(self):
        rows = [row("a", "An adult person wears a pale top in this clear portrait.",
                    {"upper_garment": {"color": "white"}},
                    {"upper_garment": {"color": "light-colored"}})]
        lexical = self.mod.score_rows(rows, self.mod.lexical_similarity)
        semantic = self.mod.score_rows(rows, lambda field, gt, pred: 1.0)
        self.assertEqual(lexical["metrics"]["micro"]["f1"], 0.0)
        self.assertEqual(semantic["metrics"]["micro"]["f1"], 1.0)

    def test_caption_metrics_use_approved_definitions(self):
        ideal = "An adult male with short dark hair wears a blue jacket and black pants with white sneakers today."
        self.assertEqual(len(ideal.split()), 18)
        rows = [
            row("a", ideal, {}, {}),
            row("b", "An adult male stands in a street. He wears blue. <pad>", {}, {}),
            row("c", "male|blue jacket|short hair", {}, {}),
            row("d", "", {}, {}),
        ]
        metrics = self.mod.score_rows(rows, self.mod.lexical_similarity)["metrics"]
        self.assertEqual(metrics["structure_pass_rate"], 0.25)
        self.assertEqual(metrics["length"]["length_18_24_ratio"], 0.25)
        self.assertEqual(metrics["background_leakage_rate"], 0.25)
        self.assertEqual(metrics["invalid_output_rate"], 0.75)

    def test_empty_attribute_denominators_are_zero(self):
        result = self.mod.score_rows([row("a", "An adult person is visible.", {}, {})], self.mod.lexical_similarity)
        self.assertEqual(result["metrics"]["micro"], {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        self.assertEqual(result["metrics"]["macro_field_f1"], 0.0)
        self.assertEqual(result["metrics"]["fabrication_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
