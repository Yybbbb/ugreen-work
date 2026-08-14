import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "archive_sft_dev_artifacts.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "archive_sft_dev_artifacts", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ArchiveDevArtifactTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run-a"
        self.run_dir.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_archive_moves_root_dev_files_into_dev_subdir(self):
        metrics = self.run_dir / "dev_metrics_step_1200.json"
        predictions = self.run_dir / "dev_predictions_step_1200.jsonl"
        nested = self.run_dir / "nested" / "dev_metrics_step_9999.json"
        other = self.run_dir / "train.log"
        self._write(metrics, '{"step": 1200}\n')
        self._write(predictions, '{"sample_id":"a"}\n')
        self._write(nested, '{"step": 9999}\n')
        self._write(other, "log\n")

        result = self.mod.archive_run_dev_artifacts(self.run_dir)

        self.assertEqual(result["moved"], 2)
        self.assertEqual(result["verified"], 0)
        self.assertFalse(metrics.exists())
        self.assertFalse(predictions.exists())
        self.assertTrue((self.run_dir / "dev" / metrics.name).is_file())
        self.assertTrue((self.run_dir / "dev" / predictions.name).is_file())
        self.assertTrue(nested.is_file())
        self.assertTrue(other.is_file())

    def test_archive_is_noop_for_identical_destination(self):
        metrics = self.run_dir / "dev_metrics_step_1200.json"
        archived = self.run_dir / "dev" / metrics.name
        self._write(metrics, '{"step": 1200}\n')
        self._write(archived, '{"step": 1200}\n')

        result = self.mod.archive_run_dev_artifacts(self.run_dir)

        self.assertEqual(result["moved"], 0)
        self.assertEqual(result["verified"], 1)
        self.assertFalse(metrics.exists())
        self.assertTrue(archived.is_file())

    def test_archive_dry_run_does_not_modify_files(self):
        metrics = self.run_dir / "dev_metrics_step_1200.json"
        self._write(metrics, '{"step": 1200}\n')

        result = self.mod.archive_run_dev_artifacts(self.run_dir, dry_run=True)

        self.assertEqual(result["planned"], 1)
        self.assertEqual(result["moved"], 0)
        self.assertTrue(metrics.is_file())
        self.assertFalse((self.run_dir / "dev").exists())
        self.assertFalse((self.run_dir / "dev" / metrics.name).exists())

    def test_archive_rejects_conflicting_destination(self):
        metrics = self.run_dir / "dev_metrics_step_1200.json"
        archived = self.run_dir / "dev" / metrics.name
        self._write(metrics, '{"step": 1200}\n')
        self._write(archived, '{"step": 9999}\n')

        with self.assertRaisesRegex(ValueError, "conflict"):
            self.mod.archive_run_dev_artifacts(self.run_dir)

        self.assertTrue(metrics.is_file())
        self.assertTrue(archived.is_file())

    def test_archive_all_runs_accumulates_impacted_run_summaries(self):
        self._write(self.run_dir / "dev_metrics_step_1200.json", '{"step": 1200}\n')
        untouched = self.root / "run-b"
        untouched.mkdir()

        result = self.mod.archive_all_runs(self.root, dry_run=True)

        self.assertEqual(result["totals"]["runs"], 1)
        self.assertEqual(result["totals"]["planned"], 1)
        self.assertEqual(
            [Path(run["run_dir"]).name for run in result["runs"]],
            ["run-a"],
        )


if __name__ == "__main__":
    unittest.main()
