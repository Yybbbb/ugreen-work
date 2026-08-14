import importlib.util
import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "scripts" / "reviewed_sft_pipeline_config.py"
PIPELINE_PATH = PROJECT_ROOT / "scripts" / "run_reviewed_sft_pipeline.py"


def load_module(name, path):
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReviewedConfigTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module("reviewed_sft_pipeline_config", CONFIG_PATH)
        self.runs = self.mod.experiment_specs(PROJECT_ROOT)

    def option(self, command, name):
        return command[command.index(name) + 1]

    def test_all_historical_run_contracts_are_exact(self):
        expected = [
            ("V1", "region_category_person_sft_30k_replay1k_b64", 64, 4, 1, 485, False, "1e-07", "5e-07", "1e-06", "0.01", "0.1", "50"),
            ("V2", "region_category_person_sft_30k_replay1k_b32", 32, 2, 1, 969, False, "1e-07", "5e-07", "1e-06", "0.01", "0.1", "100"),
            ("V3", "region_category_person_sft_30k_replay1k_b32_e1p5", 32, 2, 2, 1454, False, "1e-07", "5e-07", "1e-06", "0.01", "0.1", "100"),
            ("V4A", "region_category_person_sft_30k_replay1k_b16_e3_lr125", 16, 1, 3, 5814, False, "2e-07", "7.5e-07", "1.25e-06", "0.015", "0.03", "200"),
            ("V4B", "region_category_person_sft_30k_person_only_b16_e3_lr125", 16, 1, 3, 5625, True, "2e-07", "7.5e-07", "1.25e-06", "0.015", "0.03", "200"),
        ]
        self.assertEqual(len(self.runs), 5)
        for run, contract in zip(self.runs, expected):
            label, name, batch, accum, epochs, steps, person_only, vision, projector, language, decay, min_lr, eval_steps = contract
            command = self.mod.training_command(run)
            self.assertEqual((run.label, run.run_name), (label, name))
            self.assertEqual(self.option(command, "--formal-global-batch-size"), str(batch))
            self.assertEqual(self.option(command, "--gradient-accumulation-steps"), str(accum))
            self.assertEqual(self.option(command, "--epochs"), str(epochs))
            self.assertEqual(self.option(command, "--max-optimizer-steps"), str(steps))
            self.assertEqual(self.option(command, "--vision-lr"), vision)
            self.assertEqual(self.option(command, "--projection-lr"), projector)
            self.assertEqual(self.option(command, "--language-lr"), language)
            self.assertEqual(self.option(command, "--weight-decay"), decay)
            self.assertEqual(self.option(command, "--min-lr-ratio"), min_lr)
            self.assertEqual(self.option(command, "--eval-steps"), eval_steps)
            self.assertEqual("--person-only" in command, person_only)

    def test_common_data_and_runtime_contract(self):
        for run in self.runs:
            command = self.mod.training_command(run)
            self.assertEqual(self.option(command, "--train-data"), str(PROJECT_ROOT / "data/prepared/train.jsonl"))
            self.assertEqual(self.option(command, "--dev-data"), str(PROJECT_ROOT / "data/prepared/dev.jsonl"))
            self.assertEqual(self.option(command, "--replay-data"), str(PROJECT_ROOT / "data/prepared/native_replay.jsonl"))
            self.assertEqual(self.option(command, "--per-device-batch-size"), "4")
            self.assertEqual(self.option(command, "--label-smoothing"), "0.05")
            self.assertEqual(self.option(command, "--warmup-ratio"), "0.05")
            self.assertEqual(self.option(command, "--num-workers"), "6")
            self.assertEqual(self.option(command, "--dev-num-workers"), "0")
            self.assertEqual(self.option(command, "--prefetch-factor"), "1")
            self.assertEqual(self.mod.training_environment()["CUDA_VISIBLE_DEVICES"], "3,4,5,6")

    def test_v1_keeps_legacy_inference_path(self):
        self.assertEqual(self.runs[0].inference_dir.name, "test_inference_final_v2")
        self.assertTrue(all(run.inference_dir.name == "test_inference_final" for run in self.runs[1:]))

    def test_qwen_commands_use_dynamic_expected_sample_count(self):
        run = self.runs[0]
        stages = self.mod.qwen_stage_commands(run, expected_samples=17)
        self.assertEqual([stage.name for stage in stages], ["extract", "finalize", "qwen_gt"])
        finalize = stages[1].command
        self.assertEqual(finalize[finalize.index("--expected-samples") + 1], "17")
        self.assertIn(str(run.evaluation_dir / "metrics_qwen_gt.json"), str(stages[2].completion_path))


class ReviewedPipelineDataTests(unittest.TestCase):
    def setUp(self):
        self.config = load_module("reviewed_sft_pipeline_config", CONFIG_PATH)
        self.pipeline = load_module("run_reviewed_sft_pipeline", PIPELINE_PATH)

    @staticmethod
    def sha(path):
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def write_jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def make_prepared_root(self, root, test_rows):
        prepared = root / "data/prepared"
        manifests = root / "data/manifests"
        for split, rows in {
            "train": [{"sample_id": "tr", "label": "train"}],
            "dev": [{"sample_id": "dv", "label": "dev"}],
            "test": test_rows,
            "rl": [{"sample_id": "rl", "label": "rl"}],
        }.items():
            self.write_jsonl(prepared / f"{split}.jsonl", rows)
            manifest_rows = [{"sample_id": row["sample_id"], "caption": row["label"]} for row in rows]
            self.write_jsonl(manifests / f"{split}.jsonl", manifest_rows)
        metadata = {"splits": {}}
        for split in ("train", "dev", "test", "rl"):
            data_path = prepared / f"{split}.jsonl"
            manifest_path = manifests / f"{split}.jsonl"
            metadata["splits"][split] = {
                "path": str(data_path), "samples": len(list(data_path.open())),
                "sha256": self.sha(data_path), "manifest": str(manifest_path),
                "manifest_sha256": self.sha(manifest_path),
            }
        (prepared / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_preflight_removes_empty_test_rows_and_updates_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_prepared_root(root, [
                {"sample_id": "keep", "label": "caption"},
                {"sample_id": "drop", "label": "   "},
            ])
            result = self.pipeline.sanitize_and_validate_prepared(root)
            test_rows = list(self.pipeline.iter_jsonl(root / "data/prepared/test.jsonl"))
            manifest_rows = list(self.pipeline.iter_jsonl(root / "data/manifests/test.jsonl"))
            metadata = json.loads((root / "data/prepared/metadata.json").read_text())
        self.assertEqual(result["test_samples"], 1)
        self.assertEqual(result["removed_empty_caption_ids"], ["drop"])
        self.assertEqual([row["sample_id"] for row in test_rows], ["keep"])
        self.assertEqual([row["sample_id"] for row in manifest_rows], ["keep"])
        self.assertEqual(metadata["splits"]["test"]["samples"], 1)

    def test_archive_is_idempotent_and_never_moves_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = self.config.experiment_specs(root)
            for spec in specs:
                spec.run_dir.mkdir(parents=True)
                (spec.run_dir / "old.txt").write_text("old")
            prepared = root / "data/prepared"
            prepared.mkdir(parents=True)
            (prepared / "test_qwen_attributes.jsonl").write_text("old")
            state = {}
            state_path = root / "state.json"
            backup = self.pipeline.archive_previous_artifacts(root, specs, state, state_path, "stamp")
            specs[0].run_dir.mkdir(parents=True)
            (specs[0].run_dir / "new.txt").write_text("new")
            again = self.pipeline.archive_previous_artifacts(root, specs, state, state_path, "ignored")
            self.assertEqual(backup, again)
            self.assertTrue((backup / specs[0].run_name / "old.txt").exists())
            self.assertTrue((specs[0].run_dir / "new.txt").exists())

    def test_next_training_overlaps_previous_qwen_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.pipeline.StateStore(root / "state.json", {"stages": {}})
            runs = self.config.experiment_specs(root)
            v2_train_started = threading.Event()
            calls = []

            def fake_execute(stage, *args, **kwargs):
                calls.append((stage.experiment, stage.name))
                if stage.experiment == runs[1].run_name and stage.name == "train":
                    v2_train_started.set()
                if stage.experiment == runs[0].run_name and stage.name == "extract":
                    self.assertTrue(v2_train_started.wait(1), "V2 did not start while V1 Qwen evaluation was running")

            with mock.patch.object(self.pipeline, "wait_for_qwen"), mock.patch.object(
                self.pipeline, "execute_stage", side_effect=fake_execute
            ):
                self.pipeline.run_pipeline(root, store, 4328)

        florence = [(exp, name) for exp, name in calls if name in ("train", "infer")]
        self.assertEqual(
            florence,
            [(run.run_name, name) for run in runs for name in ("train", "infer")],
        )


if __name__ == "__main__":
    unittest.main()
