import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_person_attribute_rl.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("train_person_attribute_rl", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AlphaScheduleTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_starts_at_0_05(self):
        self.assertAlmostEqual(self.mod.alpha_at_step(0, 375), 0.10, places=6)

    def test_ends_at_0_30(self):
        self.assertAlmostEqual(self.mod.alpha_at_step(374, 375), 0.50, places=6)

    def test_midpoint_linear(self):
        self.assertAlmostEqual(self.mod.alpha_at_step(187, 375), 0.30, places=6)

    def test_clamps_above_total(self):
        self.assertAlmostEqual(self.mod.alpha_at_step(400, 375), 0.50, places=6)

    def test_accepts_explicit_endpoints(self):
        self.assertAlmostEqual(
            self.mod.alpha_at_step(9, 10, start=0.2, end=0.6), 0.6, places=6
        )

    def test_single_step_uses_final_alpha(self):
        self.assertEqual(self.mod.alpha_at_step(0, 1), 0.50)


class FormalDefaultTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_uses_v4b_final_checkpoint(self):
        self.assertEqual(
            self.mod.DEFAULT_SFT_FINAL,
            PROJECT_ROOT
            / "artifacts/sft/region_category_person_sft_30k_person_only_b16_e3_lr125/final",
        )

    def test_writes_rl_artifacts_separately(self):
        self.assertEqual(self.mod.DEFAULT_RL_OUTPUT_ROOT, PROJECT_ROOT / "artifacts/rl")

    def test_cli_exposes_formal_alpha_defaults(self):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            args = self.mod.parse_args()
        self.assertEqual(args.alpha_start, 0.10)
        self.assertEqual(args.alpha_end, 0.50)

    def test_reward_service_preflight_is_enabled_by_default(self):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            args = self.mod.parse_args()
        self.assertFalse(args.skip_reward_service_preflight)

    def test_distributed_eval_safety_defaults(self):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            args = self.mod.parse_args()
        self.assertEqual(args.ddp_timeout_minutes, 60)
        self.assertEqual(args.judge_request_concurrency, 16)


class RewardServicePreflightTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def args(self, matcher="qwen", skip=False):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            args = self.mod.parse_args()
        args.reward_matcher = matcher
        args.skip_reward_service_preflight = skip
        return args

    def test_lexical_matcher_only_checks_extractor(self):
        calls = []
        result = self.mod.run_reward_service_preflight(
            self.args("lexical"),
            preflight_fn=lambda base_url, model: calls.append((base_url, model))
            or {"model": model, "content": "OK"},
        )
        self.assertTrue(result["ok"])
        self.assertEqual([item["role"] for item in result["checks"]], ["extractor"])
        self.assertEqual(calls, [("http://127.0.0.1:6097/v1", "Qwen3.6-27B-FP8")])

    def test_qwen_matcher_checks_extractor_and_judge(self):
        calls = []
        result = self.mod.run_reward_service_preflight(
            self.args("qwen"),
            preflight_fn=lambda base_url, model: calls.append((base_url, model))
            or {"model": model, "content": "OK"},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            [item["role"] for item in result["checks"]], ["extractor", "judge"]
        )
        self.assertEqual(len(calls), 2)

    def test_skip_performs_no_http_requests(self):
        result = self.mod.run_reward_service_preflight(
            self.args(skip=True),
            preflight_fn=lambda *unused: self.fail("preflight must not be called"),
        )
        self.assertEqual(result, {"ok": True, "skipped": True, "checks": []})

    def test_failure_is_returned_for_distributed_broadcast(self):
        result = self.mod.run_reward_service_preflight(
            self.args(),
            preflight_fn=lambda *unused: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        self.assertFalse(result["ok"])
        self.assertIn("offline", result["error"])


class DevEvaluationHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_dev_artifacts_are_step_specific(self):
        run_dir = Path("/tmp/rl-run")
        self.assertEqual(
            self.mod.dev_predictions_path(run_dir, 100),
            run_dir / "dev/dev_predictions_step_100.jsonl",
        )
        self.assertEqual(
            self.mod.dev_metrics_path(run_dir, 100),
            run_dir / "dev/dev_metrics_step_100.json",
        )

    def test_summarizes_rewards_counts_and_extractor_failures(self):
        records = [
            {
                "caption": "An adult male wears a blue jacket and dark pants with white shoes while carrying a small black backpack.",
                "reward": 0.6,
                "counts": {
                    "soft_tp": 8.0,
                    "soft_fp": 2.0,
                    "soft_fn": 2.0,
                    "n_gen_assert": 10,
                    "n_fab": 1,
                },
                "extractor_failed": False,
            },
            {
                "caption": "",
                "reward": -1.0,
                "counts": {
                    "soft_tp": 0.0,
                    "soft_fp": 0.0,
                    "soft_fn": 4.0,
                    "n_gen_assert": 0,
                    "n_fab": 0,
                },
                "extractor_failed": True,
            },
        ]
        metrics = self.mod.summarize_dev_records(records, dev_loss=1.25)
        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["mean_reward"], -0.2)
        self.assertEqual(metrics["min_reward"], -1.0)
        self.assertEqual(metrics["max_reward"], 0.6)
        self.assertEqual(metrics["extractor_failures"], 1)
        self.assertAlmostEqual(metrics["mean_soft_f1"], 0.4)
        self.assertAlmostEqual(metrics["mean_fabrication_ratio"], 0.05)
        self.assertEqual(metrics["dev_loss"], 1.25)
        self.assertEqual(metrics["empty_ratio"], 0.5)

    def test_ranks_by_reward_then_loss_then_step(self):
        candidates = [
            {"name": "checkpoint-300", "mean_reward": 0.4, "dev_loss": 1.0, "global_step": 300},
            {"name": "checkpoint-200", "mean_reward": 0.5, "dev_loss": 1.2, "global_step": 200},
            {"name": "checkpoint-100", "mean_reward": 0.5, "dev_loss": 1.1, "global_step": 100},
            {"name": "checkpoint-50", "mean_reward": 0.5, "dev_loss": 1.1, "global_step": 50},
        ]
        kept, removed = self.mod.select_top_checkpoints(candidates, limit=3)
        self.assertEqual(
            [item["name"] for item in kept],
            ["checkpoint-50", "checkpoint-100", "checkpoint-200"],
        )
        self.assertEqual([item["name"] for item in removed], ["checkpoint-300"])

    def test_scores_missing_extraction_conservatively(self):
        rows = [
            {
                "sample_id": "sample-1",
                "scene": "office",
                "scale": "medium",
                "label": "An adult male wears a blue jacket.",
                "attributes": {"age_group": "adult", "gender": "male"},
            }
        ]
        records = self.mod.build_dev_reward_records(
            rows,
            ["An adult male wears a blue jacket."],
            [None],
            self.mod.lexical_similarity,
        )
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["extractor_failed"])
        self.assertEqual(records[0]["extracted_attributes"], {})
        self.assertEqual(records[0]["sample_id"], "sample-1")
        self.assertGreater(records[0]["counts"]["soft_fn"], 0.0)

    def test_scores_valid_extraction_with_selected_matcher(self):
        attrs = {"age_group": "adult", "gender": "male"}
        records = self.mod.build_dev_reward_records(
            [{"sample_id": "sample-2", "attributes": attrs, "label": "gt"}],
            ["An adult male wears a blue jacket."],
            [attrs],
            lambda field, gt, generated: 1.0,
        )
        self.assertFalse(records[0]["extractor_failed"])
        self.assertGreater(records[0]["reward"], 0.0)

    def test_batches_dev_extractor_requests(self):
        calls = []

        def fake_extract(captions, *, base_url, model):
            calls.append((list(captions), base_url, model))
            return [{"id": str(index)} for index in range(len(captions))]

        result = self.mod.extract_attributes_batched(
            ["a", "b", "c", "d", "e"],
            base_url="http://extractor/v1",
            model="extractor",
            batch_size=2,
            extract_fn=fake_extract,
        )
        self.assertEqual([len(call[0]) for call in calls], [2, 2, 1])
        self.assertEqual(len(result), 5)

    def test_enumerates_fixed_and_extra_similarity_requests(self):
        requests = self.mod.similarity_requests(
            {
                "gender": "male",
                "upper_garment": {"color": "blue"},
                "extra": ["striped shirt", "watch"],
            },
            {
                "gender": "man",
                "upper_garment": {"color": "navy"},
                "extra": ["wristwatch"],
            },
        )
        self.assertIn(("gender", "male", "male"), requests)
        self.assertIn(("upper_garment.color", "blue", "navy"), requests)
        self.assertIn(("extra", "striped shirt", "wristwatch"), requests)
        self.assertIn(("extra", "watch", "wristwatch"), requests)

    def test_round_robin_dev_shards_cover_each_original_index_once(self):
        rows = [{"sample_id": str(index)} for index in range(10)]
        shards = [
            self.mod.indexed_dev_shard(rows, rank=rank, world_size=4)
            for rank in range(4)
        ]
        self.assertEqual([index for index, unused in shards[1]], [1, 5, 9])
        self.assertEqual(
            sorted(index for shard in shards for index, unused in shard),
            list(range(10)),
        )

    def test_merges_rank_payloads_in_original_order(self):
        payloads = [
            {"ok": True, "items": [(0, {"sample_id": "0"}), (2, {"sample_id": "2"})]},
            {"ok": True, "items": [(1, {"sample_id": "1"}), (3, {"sample_id": "3"})]},
        ]
        records = self.mod.merge_dev_reward_payloads(payloads, expected_count=4)
        self.assertEqual([record["sample_id"] for record in records], ["0", "1", "2", "3"])

    def test_merge_rejects_missing_or_duplicate_indices(self):
        with self.assertRaisesRegex(RuntimeError, "indices"):
            self.mod.merge_dev_reward_payloads(
                [{"ok": True, "items": [(0, {}), (0, {})]}], expected_count=2
            )

    def test_merge_propagates_rank_error(self):
        with self.assertRaisesRegex(RuntimeError, "rank 2.*offline"):
            self.mod.merge_dev_reward_payloads(
                [{"ok": False, "rank": 2, "error": "offline"}], expected_count=1
            )


class ResumeHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_resume_infers_original_run_directory(self):
        output_root = Path("/tmp/artifacts/rl")
        checkpoint = output_root / "run-a/checkpoint-100"
        self.assertEqual(
            self.mod.resolve_run_dir(output_root, None, checkpoint, "unused"),
            output_root / "run-a",
        )

    def test_resume_accepts_matching_explicit_run_name(self):
        output_root = Path("/tmp/artifacts/rl")
        checkpoint = output_root / "run-a/final"
        self.assertEqual(
            self.mod.resolve_run_dir(output_root, "run-a", checkpoint, "unused"),
            output_root / "run-a",
        )

    def test_resume_rejects_different_explicit_run_name(self):
        output_root = Path("/tmp/artifacts/rl")
        checkpoint = output_root / "run-a/checkpoint-100"
        with self.assertRaisesRegex(ValueError, "original run directory"):
            self.mod.resolve_run_dir(output_root, "run-b", checkpoint, "unused")

    def test_new_run_uses_generated_name(self):
        self.assertEqual(
            self.mod.resolve_run_dir(Path("/tmp/rl"), None, None, "generated"),
            Path("/tmp/rl/generated"),
        )

    def test_final_status_distinguishes_complete_and_capped(self):
        self.assertEqual(
            self.mod.final_status(global_step=748, total_steps=748, max_optimizer_steps=0),
            {"completed": True, "stopped_by_max_steps": False},
        )
        self.assertEqual(
            self.mod.final_status(global_step=10, total_steps=748, max_optimizer_steps=10),
            {"completed": False, "stopped_by_max_steps": True},
        )

    def test_invariants_capture_reward_and_resume_contract(self):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            args = self.mod.parse_args()
        invariants = self.mod._invariants(
            args,
            data_metadata={"train_samples": 5981},
            world_size=4,
            steps_per_epoch=748,
            total_steps=748,
        )
        self.assertEqual(invariants["model_source"], str(args.model_path.resolve()))
        self.assertEqual(invariants["alpha"], {"start": 0.10, "end": 0.50})
        self.assertEqual(invariants["reward"]["matcher"], "qwen")
        self.assertEqual(invariants["reward"]["extractor_base_url"], "http://127.0.0.1:6097/v1")
        self.assertEqual(invariants["reward"]["judge_base_url"], "http://127.0.0.1:6098/v1")
        self.assertEqual(invariants["evaluation"]["generation_samples"], 128)
        self.assertEqual(invariants["evaluation"]["reward_batch_size"], 4)
        self.assertEqual(invariants["steps_per_epoch"], 748)
        self.assertFalse(invariants["reward"]["service_preflight_skipped"])
        self.assertEqual(
            invariants["reward"]["services"]["extractor"]["max_num_batched_tokens"],
            8192,
        )
        self.assertEqual(
            invariants["reward"]["services"]["judge"]["max_num_seqs"], 128
        )
        self.assertEqual(invariants["distributed"]["timeout_minutes"], 60)
        self.assertEqual(invariants["reward"]["judge_request_concurrency"], 16)


class AdvantageTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_sampled_better_positive(self):
        self.assertAlmostEqual(self.mod.advantage(0.5, 0.3), 0.2, places=6)

    def test_sampled_worse_negative(self):
        self.assertAlmostEqual(self.mod.advantage(0.1, 0.4), -0.3, places=6)


class ScstLossTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_reinforce_formula(self):
        # L = -mean(advantage * logprob)
        # advantages [0.5, 0.5], logprobs [-1.0, -2.0] -> -mean(-0.5, -1.0) = 0.75
        self.assertAlmostEqual(self.mod.scst_loss([0.5, 0.5], [-1.0, -2.0]), 0.75, places=6)

    def test_zero_advantage_zero_loss(self):
        self.assertAlmostEqual(self.mod.scst_loss([0.0, 0.0], [-1.0, -2.0]), 0.0, places=6)


class MixedLossTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_weighted_combination(self):
        # (1-0.1)*1.0 + 0.1*2.0 = 0.9 + 0.2 = 1.1
        self.assertAlmostEqual(self.mod.mixed_loss(1.0, 2.0, 0.1), 1.1, places=6)

    def test_alpha_zero_is_pure_ce(self):
        self.assertAlmostEqual(self.mod.mixed_loss(1.0, 5.0, 0.0), 1.0, places=6)


class RlOptimizerStepsTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_6000_samples_global_batch_16(self):
        # 6000 / 16 = 375
        self.assertEqual(self.mod.rl_optimizer_steps(6000, world_size=4, per_device_batch=4, accumulation=1, epochs=1), 375)

    def test_reviewed_5981_samples_global_batch_8(self):
        self.assertEqual(self.mod.FORMAL_RL_SAMPLES, 5981)
        self.assertEqual(
            self.mod.rl_optimizer_steps(
                5981, world_size=4, per_device_batch=2, accumulation=1, epochs=1
            ),
            748,
        )


if __name__ == "__main__":
    unittest.main()
