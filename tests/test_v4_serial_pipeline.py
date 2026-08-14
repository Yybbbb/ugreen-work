import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "scripts" / "v4_serial_config.py"
PIPELINE_PATH = PROJECT_ROOT / "scripts" / "run_v4_serial_pipeline.py"


def load_module(name, path):
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class V4ConfigTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module("v4_serial_config", CONFIG_PATH)
        self.replay, self.person_only = self.mod.experiment_specs(PROJECT_ROOT)

    def test_replay_run_contract(self):
        self.assertEqual(self.replay.total_samples, 31000)
        self.assertEqual(self.replay.optimizer_steps, 5814)
        self.assertFalse(self.replay.person_only)
        command = self.mod.training_command(self.replay)
        self.assertNotIn("--person-only", command)
        self.assertEqual(command[command.index("--formal-global-batch-size") + 1], "16")
        self.assertEqual(command[command.index("--max-optimizer-steps") + 1], "5814")

    def test_person_only_run_contract(self):
        self.assertEqual(self.person_only.total_samples, 30000)
        self.assertEqual(self.person_only.optimizer_steps, 5625)
        self.assertTrue(self.person_only.person_only)
        command = self.mod.training_command(self.person_only)
        self.assertIn("--person-only", command)
        self.assertEqual(command[command.index("--max-optimizer-steps") + 1], "5625")

    def test_common_aggressive_hyperparameters(self):
        for experiment in (self.replay, self.person_only):
            command = self.mod.training_command(experiment)
            expected = {
                "--epochs": "3",
                "--per-device-batch-size": "4",
                "--gradient-accumulation-steps": "1",
                "--vision-lr": "2e-07",
                "--projection-lr": "7.5e-07",
                "--language-lr": "1.25e-06",
                "--weight-decay": "0.015",
                "--label-smoothing": "0.05",
                "--warmup-ratio": "0.05",
                "--min-lr-ratio": "0.03",
                "--eval-steps": "200",
            }
            for option, value in expected.items():
                self.assertEqual(command[command.index(option) + 1], value)
            self.assertEqual(command[command.index("--model-path") + 1], str(PROJECT_ROOT / "pretrained" / "Florence-2-base"))

    def test_stage_order_is_serial_per_experiment(self):
        self.assertEqual(
            [stage.name for stage in self.mod.stage_specs(self.replay)],
            ["train", "infer", "extract", "finalize", "document"],
        )
        ordered = self.mod.serial_stage_specs(PROJECT_ROOT)
        self.assertEqual(
            [(stage.experiment, stage.name) for stage in ordered],
            [(self.replay.run_name, name) for name in ("train", "infer", "extract", "finalize", "document")]
            + [(self.person_only.run_name, name) for name in ("train", "infer", "extract", "finalize", "document")],
        )


class V4ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_module("v4_serial_config", CONFIG_PATH)
        self.pipeline = load_module("run_v4_serial_pipeline", PIPELINE_PATH)

    def _stage(self, root, name, completion=None):
        return self.config.StageSpec(
            experiment="run-a",
            name=name,
            command=("command", name),
            environment={},
            completion_path=Path(completion) if completion else None,
            log_path=Path(root) / f"{name}.log",
        )

    def test_completed_artifact_skips_stage(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            done = Path(directory) / "done"
            done.write_text("ok", encoding="utf-8")
            stage = self._stage(directory, "train", done)
            calls = []
            self.pipeline.execute_stages(
                [stage], {}, runner=lambda *args, **kwargs: calls.append(args)
            )
        self.assertEqual(calls, [])

    def test_failed_stage_stops_before_later_stage(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            stages = [self._stage(directory, "train"), self._stage(directory, "infer")]
            calls = []

            def fail_first(command, **kwargs):
                calls.append(tuple(command))
                raise subprocess.CalledProcessError(3, command)

            with self.assertRaises(subprocess.CalledProcessError):
                self.pipeline.execute_stages(stages, {}, runner=fail_first)
        self.assertEqual(calls, [("command", "train")])


if __name__ == "__main__":
    unittest.main()
