import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_person_attribute_rl.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("evaluate_person_attribute_rl", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PredictionValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_validation_preserves_test_order_and_requires_frozen_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            test = root / "test.jsonl"
            predictions = root / "predictions.jsonl"
            source = [
                {"sample_id": "a", "attributes": {"gender": "male"}, "session": "s1"},
                {"sample_id": "b", "attributes": {"gender": "female"}, "session": "s2"},
            ]
            write_jsonl(test, source)
            generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": 64}
            write_jsonl(predictions, [
                {"sample_id": "a", "prediction": "An adult male is visible.", "checkpoint": str(checkpoint.resolve()), "generation": generation},
                {"sample_id": "b", "prediction": "An adult female is visible.", "checkpoint": str(checkpoint.resolve()), "generation": generation},
            ])
            rows = self.mod.validate_predictions(test, predictions, checkpoint, expected_samples=2)
            self.assertEqual([item["sample_id"] for item in rows], ["a", "b"])
            self.assertEqual(rows[0]["ground_truth"], {"gender": "male"})

    def test_validation_rejects_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            test = root / "test.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(test, [{"sample_id": "a", "attributes": {}}])
            write_jsonl(predictions, [{
                "sample_id": "a", "prediction": "x", "checkpoint": "/wrong",
                "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 64},
            }])
            with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                self.mod.validate_predictions(test, predictions, checkpoint, expected_samples=1)

    def test_validation_can_limit_a_full_file_for_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            test = root / "test.jsonl"
            predictions = root / "predictions.jsonl"
            generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": 64}
            write_jsonl(test, [{"sample_id": str(index), "attributes": {}} for index in range(3)])
            write_jsonl(predictions, [{
                "sample_id": str(index), "prediction": "An adult person is visible.",
                "checkpoint": str(checkpoint.resolve()), "generation": generation,
            } for index in range(3)])
            rows = self.mod.validate_predictions(
                test, predictions, checkpoint, expected_samples=2, max_samples=2
            )
            self.assertEqual([item["sample_id"] for item in rows], ["0", "1"])


class ResumeAndJudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_extraction_resumes_successes_and_writes_original_order(self):
        rows = [
            {"sample_id": "a", "prediction": "A", "ground_truth": {}, "session": "s"},
            {"sample_id": "b", "prediction": "B", "ground_truth": {}, "session": "s"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "journal.jsonl"
            final = root / "extractions.jsonl"
            write_jsonl(journal, [{"id": "a", "sample_id": "a", "attributes": {"gender": "male"}, "error": None}])
            calls = []

            def request(batch, **kwargs):
                calls.extend(item["sample_id"] for item in batch)
                return [{"id": item["sample_id"], "attributes": {"gender": "female"}, "raw": "{}", "error": None} for item in batch]

            extracted = self.mod.extract_candidate_rows(
                rows, journal, final, request_fn=request, base_url="http://extractor/v1",
                model="extractor", batch_size=8, workers=2, timeout=1, retries=0,
                max_tokens=32,
            )
            self.assertEqual(calls, ["b"])
            self.assertEqual([item["sample_id"] for item in extracted], ["a", "b"])
            self.assertEqual([item["sample_id"] for item in self.mod.iter_jsonl(final)], ["a", "b"])

    def test_extraction_refuses_unresolved_failure(self):
        rows = [{"sample_id": "a", "prediction": "A", "ground_truth": {}, "session": "s"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fail(batch, **kwargs):
                return [{"id": "a", "attributes": None, "raw": "", "error": "timeout"}]

            with self.assertRaisesRegex(RuntimeError, "unresolved extraction"):
                self.mod.extract_candidate_rows(
                    rows, root / "journal.jsonl", root / "final.jsonl", request_fn=fail,
                    base_url="x", model="x", batch_size=1, workers=1, timeout=1,
                    retries=0, max_tokens=8,
                )

    def test_persistent_judge_prefetches_only_missing_unique_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "judge.json"
            calls = []

            class FakeJudge:
                def prefetch(self, requests):
                    calls.extend(requests)

                def __call__(self, field, gt, pred):
                    return 0.75

            judge = self.mod.PersistentJudge(cache_path, FakeJudge())
            requests = [("gender", "male", "man"), ("gender", "male", "man"), ("color", "blue", "navy")]
            judge.prefetch(requests)
            self.assertEqual(len(calls), 2)
            self.assertEqual(judge("gender", "male", "man"), 0.75)
            reloaded = self.mod.PersistentJudge(cache_path, FakeJudge())
            reloaded.prefetch(requests)
            self.assertEqual(len(calls), 2)


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_publish_requires_exact_three_candidates_and_writes_reports(self):
        scorer = load_module()
        metric_mod = sys.modules["rl_test_metrics"]
        base_row = [{
            "sample_id": "a", "session": "s", "prediction": "An adult male is visible today.",
            "ground_truth": {"gender": "male"}, "attributes": {"gender": "male"},
        }]
        candidates = [
            metric_mod.build_candidate(name, f"/{name}", base_row, metric_mod.lexical_similarity)
            for name in ("v4b_sft", "qwen_rl", "lexical_rl")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = scorer.publish_reports(candidates, root, bootstrap_replicates=10)
            self.assertEqual(result["recommendation"]["candidate"], "v4b_sft")
            self.assertTrue((root / "person_caption_metrics.json").is_file())
            self.assertTrue((root / "person_caption_comparison.md").is_file())
            with self.assertRaisesRegex(ValueError, "exactly"):
                scorer.publish_reports(candidates[:2], root, bootstrap_replicates=10)

    def test_frozen_run_config_rejects_resume_with_changed_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_config.json"
            self.mod.ensure_frozen_config(path, {"extractor_model": "27b", "test_sha256": "abc"})
            self.mod.ensure_frozen_config(path, {"extractor_model": "27b", "test_sha256": "abc"})
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                self.mod.ensure_frozen_config(path, {"extractor_model": "other", "test_sha256": "abc"})


class StreamingPipelineTests(unittest.TestCase):
    """Batch-pipelined extract -> judge across all candidates in one stream."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_stream_preserves_order_resumes_journal_and_writes_finals(self):
        rows0 = [
            {"sample_id": "a", "prediction": "An adult male.", "ground_truth": {"gender": "male"}, "session": "s"},
            {"sample_id": "b", "prediction": "An adult female.", "ground_truth": {"gender": "female"}, "session": "s"},
        ]
        rows1 = [
            {"sample_id": "a", "prediction": "A young male.", "ground_truth": {"gender": "male"}, "session": "s"},
            {"sample_id": "b", "prediction": "A young female.", "ground_truth": {"gender": "female"}, "session": "s"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "eval"
            # pre-seed candidate 0 journal so sample "a" is already extracted (resume)
            seeded_dir = out / "c0"
            seeded_dir.mkdir(parents=True)
            write_jsonl(seeded_dir / "extraction.journal.jsonl", [{
                "id": "a", "sample_id": "a", "scene": None, "session": "s",
                "prediction": "An adult male.", "ground_truth": {"gender": "male"},
                "attributes": {"gender": "male"}, "raw": "", "error": None,
            }])
            extracted_ids = []

            def request(batch, **kwargs):
                extracted_ids.extend(item["sample_id"] for item in batch)
                return [{"id": item["sample_id"], "attributes": {"gender": "male"},
                         "raw": "{}", "error": None} for item in batch]

            class FakeJudge:
                def prefetch(self, requests):
                    pass

                def __call__(self, field, gt, pred):
                    return 1.0

            prepared = self.mod.stream_extract_and_judge(
                [("c0", "/c0", rows0), ("c1", "/c1", rows1)], out,
                request_fn=request, base_url="x", model="x", batch_size=8, workers=2,
                timeout=1, retries=0, max_tokens=8, judge=FakeJudge(),
            )
            # resumed sample "a" of c0 is NOT re-extracted; c1 has no journal so both extracted
            self.assertEqual(sorted(extracted_ids), ["a", "b", "b"])
            names = [name for name, _cp, _rows in prepared]
            self.assertEqual(names, ["c0", "c1"])
            self.assertEqual([r["sample_id"] for r in prepared[0][2]], ["a", "b"])
            self.assertEqual([r["sample_id"] for r in prepared[1][2]], ["a", "b"])
            # final extractions written in test order
            self.assertEqual(
                [r["sample_id"] for r in self.mod.iter_jsonl(out / "c0" / "extractions.jsonl")],
                ["a", "b"],
            )

    def test_stream_prefetches_judge_per_completed_batch(self):
        rows = [
            {"sample_id": "a", "prediction": "An adult male wears red.",
             "ground_truth": {"gender": "male", "upper_garment.color": "red"}, "session": "s"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eval"

            def request(batch, **kwargs):
                return [{"id": item["sample_id"],
                         "attributes": {"gender": "male", "upper_garment.color": "red"},
                         "raw": "{}", "error": None} for item in batch]

            prefetched = []

            class FakeJudge:
                def prefetch(self, requests):
                    prefetched.extend(requests)

                def __call__(self, field, gt, pred):
                    return 1.0

            self.mod.stream_extract_and_judge(
                [("c0", "/c0", rows)], out, request_fn=request, base_url="x", model="x",
                batch_size=8, workers=1, timeout=1, retries=0, max_tokens=8, judge=FakeJudge(),
            )
            self.assertTrue(prefetched)
            fields = {req[0] for req in prefetched}
            self.assertIn("gender", fields)
            # every prefetched item is a (field, gt, pred) triple of strings
            for item in prefetched:
                self.assertEqual(len(item), 3)

    def test_stream_refuses_unresolved_extraction_failure(self):
        rows = [{"sample_id": "a", "prediction": "x", "ground_truth": {}, "session": "s"}]
        with tempfile.TemporaryDirectory() as tmp:

            def fail(batch, **kwargs):
                return [{"id": "a", "attributes": None, "raw": "", "error": "timeout"}]

            class FakeJudge:
                def prefetch(self, requests):
                    pass

                def __call__(self, field, gt, pred):
                    return 0.0

            with self.assertRaisesRegex(RuntimeError, "unresolved extraction"):
                self.mod.stream_extract_and_judge(
                    [("c0", "/c0", rows)], Path(tmp) / "eval", request_fn=fail,
                    base_url="x", model="x", batch_size=1, workers=1, timeout=1,
                    retries=0, max_tokens=8, judge=FakeJudge(),
                )


if __name__ == "__main__":
    unittest.main()
