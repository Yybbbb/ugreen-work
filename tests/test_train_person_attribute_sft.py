import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_person_attribute_sft.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "train_person_attribute_sft", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StepMathTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_optimizer_steps_matches_31k_four_gpu_plan(self):
        self.assertEqual(
            self.mod.optimizer_steps(
                31000, world_size=4, per_device_batch=4, accumulation=4
            ),
            485,
        )

    def test_scheduler_steps_follow_explicit_v3_step_cap(self):
        self.assertEqual(self.mod.scheduler_total_steps(969, 2, 1454), 1454)
        self.assertEqual(self.mod.scheduler_total_steps(969, 1, 0), 969)

    def test_parser_defaults_match_four_gpu_low_lr_plan(self):
        with patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            args = self.mod.parse_args()

        self.assertEqual(args.per_device_batch_size, 4)
        self.assertEqual(args.gradient_accumulation_steps, 4)
        self.assertEqual(args.vision_lr, 1e-7)
        self.assertEqual(args.projection_lr, 5e-7)
        self.assertEqual(args.language_lr, 1e-6)
        self.assertEqual(args.warmup_ratio, 0.05)
        self.assertEqual(args.min_lr_ratio, 0.1)
        self.assertEqual(args.eval_steps, 50)
        self.assertEqual(args.num_workers, 6)
        self.assertEqual(args.dev_num_workers, 0)
        self.assertEqual(args.prefetch_factor, 1)

    def test_cosine_floor_keeps_low_learning_rate_tail(self):
        self.assertAlmostEqual(
            self.mod.cosine_with_floor_multiplier(
                current_step=100, warmup_steps=5, total_steps=100, min_lr_ratio=0.1
            ),
            0.1,
        )
        self.assertAlmostEqual(
            self.mod.cosine_with_floor_multiplier(
                current_step=5, warmup_steps=5, total_steps=100, min_lr_ratio=0.1
            ),
            1.0,
        )

    def test_completed_resume_has_no_remaining_epochs(self):
        self.assertFalse(self.mod.has_remaining_epochs(start_epoch=1, total_epochs=1))
        self.assertTrue(self.mod.has_remaining_epochs(start_epoch=0, total_epochs=1))

    def test_formal_training_requires_four_processes(self):
        with self.assertRaisesRegex(ValueError, "WORLD_SIZE=4"):
            self.mod.validate_world_size(world_size=1, formal=True)
        self.mod.validate_world_size(world_size=4, formal=True)
        self.mod.validate_world_size(world_size=1, formal=False)

    def test_formal_training_requires_global_batch_64(self):
        self.mod.validate_global_batch(4, 4, 4, formal=True)
        self.mod.validate_global_batch(4, 2, 8, formal=True)
        with self.assertRaisesRegex(ValueError, "global batch 64"):
            self.mod.validate_global_batch(4, 2, 4, formal=True)
        self.mod.validate_global_batch(1, 4, 1, formal=False)

    def test_formal_batch_can_be_explicitly_set_for_v2(self):
        self.mod.validate_global_batch(4, 4, 2, formal=True, expected_global_batch=32)
        with self.assertRaisesRegex(ValueError, "global batch 32"):
            self.mod.validate_global_batch(4, 4, 4, formal=True, expected_global_batch=32)

    def test_tail_accumulation_group_uses_actual_size(self):
        self.assertEqual(self.mod.accumulation_group_size(3874, 3875, 4), 3)
        self.assertEqual(self.mod.accumulation_group_size(12, 20, 4), 4)

    def test_resume_selects_rng_state_for_current_rank(self):
        state = {"rng_by_rank": [{"value": "rank-0"}, {"value": "rank-1"}]}
        self.assertEqual(
            self.mod.rng_state_for_rank(state, rank=1, world_size=2),
            {"value": "rank-1"},
        )

    def test_resume_rejects_rng_world_size_mismatch(self):
        state = {"rng_by_rank": [{"value": "rank-0"}]}
        with self.assertRaisesRegex(ValueError, "RNG state count"):
            self.mod.rng_state_for_rank(state, rank=0, world_size=2)


class ParameterRoutingTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_parameter_roles_are_mutually_defined(self):
        self.assertEqual(
            self.mod.parameter_role("vision_tower.blocks.0.weight"), "vision"
        )
        self.assertEqual(self.mod.parameter_role("image_projection"), "projection")
        self.assertEqual(
            self.mod.parameter_role("image_proj_norm.weight"), "projection"
        )
        self.assertEqual(
            self.mod.parameter_role("language_model.encoder.layers.0.weight"),
            "language",
        )

    def test_bias_and_norm_do_not_decay(self):
        self.assertTrue(self.mod.is_no_decay_parameter("layer.self_attn.bias"))
        self.assertTrue(self.mod.is_no_decay_parameter("image_proj_norm.weight"))
        self.assertFalse(self.mod.is_no_decay_parameter("layer.fc1.weight"))


class DataAndMetricTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_mixing_is_deterministic_and_keeps_all_rows(self):
        person = [
            {"sample_id": "person-a", "task": "<REGION_TO_CATEGORY>"},
            {"sample_id": "person-b", "task": "<REGION_TO_CATEGORY>"},
        ]
        replay = [
            {"sample_id": "replay-a", "task": "<REGION_TO_DESCRIPTION>"}
        ]
        first = self.mod.mix_training_rows(person, replay, seed=17)
        second = self.mod.mix_training_rows(reversed(person), replay, seed=17)

        self.assertEqual(first, second)
        self.assertEqual({row["sample_id"] for row in first}, {"person-a", "person-b", "replay-a"})

    def test_caption_metrics_report_length_sentence_and_subject(self):
        captions = [
            "An adult female with long black hair wears a white jacket, black pants, glasses, and white shoes while carrying a backpack.",
            "blue jacket, black pants",
            "",
        ]

        metrics = self.mod.caption_metrics(captions)

        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["empty_outputs"], 1)
        self.assertAlmostEqual(metrics["single_sentence_ratio"], 1 / 3)
        self.assertAlmostEqual(metrics["subject_start_ratio"], 1 / 3)
        self.assertAlmostEqual(metrics["length_18_24_ratio"], 1 / 3)


class DevArtifactPathTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_dev_artifact_paths_live_under_dev_subdirectory(self):
        run_dir = Path("/tmp/example-run")

        self.assertEqual(self.mod.dev_artifact_dir(run_dir), run_dir / "dev")
        self.assertEqual(
            self.mod.dev_metrics_path(run_dir, 1200),
            run_dir / "dev" / "dev_metrics_step_1200.json",
        )
        self.assertEqual(
            self.mod.dev_predictions_path(run_dir, 1200),
            run_dir / "dev" / "dev_predictions_step_1200.jsonl",
        )


class IndexedJsonlRowsTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def _row(self, sample_id, task):
        bbox = [1, 2, 3, 4]
        return {
            "sample_id": sample_id,
            "task": task,
            "prompt": self.mod.build_region_prompt(task, bbox),
            "label": "An adult person wearing a dark jacket.",
            "image": "/tmp/person.jpg",
            "bbox_loc_0_999": bbox,
        }

    def test_index_reads_rows_on_demand_without_storing_row_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = [
                self._row("person-a", self.mod.PERSON_TASK),
                self._row("person-b", self.mod.PERSON_TASK),
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            indexed = self.mod.IndexedJsonlRows([(path, self.mod.PERSON_TASK)])

            self.assertEqual(len(indexed), 2)
            self.assertEqual(indexed[1]["sample_id"], "person-b")
            self.assertNotIn("_rows", indexed.__dict__)
            self.assertEqual(len(indexed.locations), 2)
            self.assertEqual(indexed.source_counts[path.resolve()], 2)

    def test_index_supports_per_source_limits_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            person_path = directory / "person.jsonl"
            replay_path = directory / "replay.jsonl"
            person_rows = [
                self._row("person-a", self.mod.PERSON_TASK),
                self._row("person-b", self.mod.PERSON_TASK),
            ]
            replay_rows = [self._row("replay-a", self.mod.REPLAY_TASK)]
            person_path.write_text(
                "".join(json.dumps(row) + "\n" for row in person_rows), encoding="utf-8"
            )
            replay_path.write_text(
                "".join(json.dumps(row) + "\n" for row in replay_rows), encoding="utf-8"
            )

            indexed = self.mod.IndexedJsonlRows(
                [
                    (person_path, self.mod.PERSON_TASK, 1),
                    (replay_path, self.mod.REPLAY_TASK, 1),
                ]
            )
            self.assertEqual(
                [indexed[index]["sample_id"] for index in range(len(indexed))],
                ["person-a", "replay-a"],
            )

            duplicate_path = directory / "duplicate.jsonl"
            duplicate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in [
                    self._row("same", self.mod.PERSON_TASK),
                    self._row("same", self.mod.PERSON_TASK),
                ]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                self.mod.IndexedJsonlRows([(duplicate_path, self.mod.PERSON_TASK)])

    def test_index_allows_same_sample_id_for_different_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            person_path = directory / "person.jsonl"
            replay_path = directory / "replay.jsonl"
            person_path.write_text(
                json.dumps(self._row("shared", self.mod.PERSON_TASK)) + "\n",
                encoding="utf-8",
            )
            replay_path.write_text(
                json.dumps(self._row("shared", self.mod.REPLAY_TASK)) + "\n",
                encoding="utf-8",
            )

            indexed = self.mod.IndexedJsonlRows([
                (person_path, self.mod.PERSON_TASK),
                (replay_path, self.mod.REPLAY_TASK),
            ])

        self.assertEqual(len(indexed), 2)


if __name__ == "__main__":
    unittest.main()
