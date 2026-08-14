import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "update_v4_training_runs.py"


def load_module():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("update_v4_training_runs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V4DocumentTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def _metrics(self, f1):
        return {
            "samples": 4328,
            "extractor_failures": 0,
            "micro": {"precision": f1 + 0.05, "recall": f1 - 0.05, "f1": f1},
            "soft_micro": {"f1": f1 + 0.10},
            "macro_field_f1": f1 - 0.10,
            "mean_field_exact": f1 - 0.05,
            "unknown_extra_rate": 0.09,
            "caption": {
                "average_words": 21.0,
                "length_18_24_ratio": 0.55,
                "background_keyword_ratio": 0.001,
            },
        }

    def test_pending_render_contains_both_serial_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            block = self.mod.render_v4_block(Path(directory))
        self.assertIn("V4 replay1k", block)
        self.assertIn("V4 no-replay", block)
        self.assertEqual(block.count("等待串行执行"), 2)

    def test_partial_metrics_render_completed_first_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay, _ = self.mod.experiment_specs(root)
            path = replay.inference_dir / "evaluation_qwen_final" / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(self._metrics(0.61)), encoding="utf-8")
            block = self.mod.render_v4_block(root)
        self.assertIn("0.6100", block)
        self.assertIn("评估完成", block)
        self.assertIn("等待串行执行", block)

    def test_marker_update_preserves_surrounding_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.md"
            path.write_text(
                "before\n<!-- V4_RESULTS_START -->\nold\n<!-- V4_RESULTS_END -->\nafter\n",
                encoding="utf-8",
            )
            self.mod.update_marker_block(path, "new")
            content = path.read_text(encoding="utf-8")
        self.assertIn("before", content)
        self.assertIn("after", content)
        self.assertIn("\nnew\n", content)
        self.assertNotIn("old", content)


if __name__ == "__main__":
    unittest.main()
