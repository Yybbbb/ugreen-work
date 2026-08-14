import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_person_attribute_rl_evaluation.sh"


class EvaluationRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT_PATH.read_text(encoding="utf-8")

    def test_waits_for_lexical_final_and_detects_dead_training(self):
        self.assertIn('LEXICAL_CHECKPOINT/model.safetensors', self.text)
        self.assertIn("pgrep -f", self.text)
        self.assertIn("train_person_attribute_rl.py.*region_category_person_scst_reviewed5981_lexical", self.text)

    def test_uses_locked_models_and_endpoints(self):
        for value in (
            "Qwen3.6-27B-FP8", "http://127.0.0.1:6097/v1",
            "Qwen3.5-4B", "http://127.0.0.1:6098/v1",
        ):
            self.assertIn(value, self.text)

    def test_runs_rl_inference_serially_on_gpu_3_to_6(self):
        self.assertIn("CUDA_VISIBLE_DEVICES=3,4,5,6", self.text)
        self.assertEqual(self.text.count("scripts/infer_person_attribute.py"), 2)
        qwen_position = self.text.index('"$QWEN_CHECKPOINT"')
        lexical_position = self.text.index('"$LEXICAL_CHECKPOINT"')
        self.assertLess(qwen_position, lexical_position)

    def test_evaluates_exact_three_named_candidates(self):
        for name in ("v4b_sft|", "qwen_rl|", "lexical_rl|"):
            self.assertIn(name, self.text)
        self.assertIn("scripts/evaluate_person_attribute_rl.py", self.text)
        self.assertIn("artifacts/rl/evaluation", self.text)

    def test_logs_and_writes_completion_marker(self):
        self.assertIn("runner.log", self.text)
        self.assertIn("runner.pid", self.text)
        self.assertIn("evaluation.complete", self.text)
        self.assertIn("evaluation.failed", self.text)


if __name__ == "__main__":
    unittest.main()
