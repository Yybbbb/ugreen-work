import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "rl_clients.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("rl_clients", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ParseJudgeContentTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_score_one(self):
        self.assertEqual(self.mod.parse_judge_content('{"score":1.0,"reason":"same"}'), 1.0)

    def test_score_zero(self):
        self.assertEqual(self.mod.parse_judge_content('{"score":0.0,"reason":"red vs blue"}'), 0.0)

    def test_score_half(self):
        self.assertEqual(self.mod.parse_judge_content('{"score":0.5}'), 0.5)

    def test_score_0_75(self):
        self.assertEqual(self.mod.parse_judge_content('{"score":0.75}'), 0.75)

    def test_markdown_fenced(self):
        self.assertEqual(self.mod.parse_judge_content('```json\n{"score":0.25}\n```'), 0.25)

    def test_garbage_defaults_zero(self):
        # conservative: unparseable judge output -> 0.0 (conflict)
        self.assertEqual(self.mod.parse_judge_content("I cannot decide"), 0.0)

    def test_empty_defaults_zero(self):
        self.assertEqual(self.mod.parse_judge_content(""), 0.0)


class ParseExtractorContentTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_results_wrapper(self):
        text = '{"results":[{"id":"0","gender":"male"}]}'
        out = self.mod.parse_extractor_content(text)
        self.assertEqual(out[0]["id"], "0")
        self.assertEqual(out[0]["gender"], "male")

    def test_single_object(self):
        out = self.mod.parse_extractor_content('{"id":"0","gender":"male"}')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["gender"], "male")

    def test_invalid_returns_empty(self):
        self.assertEqual(self.mod.parse_extractor_content("not json"), [])


class ServicePreflightTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_parses_model_ids(self):
        envelope = {"object": "list", "data": [{"id": "Qwen3.6-27B-FP8"}, {"id": "other"}]}
        self.assertEqual(
            self.mod.parse_model_ids(envelope),
            ["Qwen3.6-27B-FP8", "other"],
        )

    def test_accepts_ok_with_null_reasoning(self):
        self.mod.validate_probe_message(
            {"content": " OK\n", "reasoning": None, "reasoning_content": None}
        )

    def test_rejects_non_ok_content(self):
        with self.assertRaisesRegex(RuntimeError, "expected content 'OK'"):
            self.mod.validate_probe_message({"content": "not ok", "reasoning": None})

    def test_rejects_nonempty_reasoning(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected reasoning"):
            self.mod.validate_probe_message({"content": "OK", "reasoning": "thinking"})

    def test_preflight_rejects_wrong_served_model(self):
        with mock.patch.object(
            self.mod, "_get_json", return_value={"data": [{"id": "wrong"}]}
        ):
            with self.assertRaisesRegex(RuntimeError, "expected model"):
                self.mod.preflight_service("http://127.0.0.1:6097/v1", "expected")

    def test_preflight_checks_model_and_ok_probe(self):
        with mock.patch.object(
            self.mod,
            "_get_json",
            return_value={"data": [{"id": "Qwen3.6-27B-FP8"}]},
        ) as get_json, mock.patch.object(
            self.mod,
            "_post_chat_message",
            return_value={"content": "OK", "reasoning": None},
        ) as post_message:
            result = self.mod.preflight_service(
                "http://127.0.0.1:6097/v1", "Qwen3.6-27B-FP8"
            )
        self.assertEqual(result["model"], "Qwen3.6-27B-FP8")
        self.assertEqual(result["content"], "OK")
        get_json.assert_called_once()
        kwargs = post_message.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["top_p"], 1.0)


class JudgePrefetchTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_prefetch_deduplicates_runs_concurrently_and_populates_cache(self):
        lock = threading.Lock()
        active = 0
        max_active = 0
        calls = []

        def fake_post(**kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                calls.append(kwargs["messages"][-1]["content"])
            time.sleep(0.03)
            with lock:
                active -= 1
            return '{"score": 1.0}'

        with mock.patch.object(self.mod, "_post_chat", side_effect=fake_post):
            judge = self.mod.make_judge_fn(
                "http://judge/v1", "judge", concurrency=4
            )
            judge.prefetch(
                [
                    ("gender", "male", "man"),
                    ("gender", "male", "man"),
                    ("upper.color", "blue", "navy"),
                    ("lower.color", "black", "dark"),
                ]
            )
            self.assertEqual(judge("gender", "male", "man"), 1.0)

        self.assertEqual(len(calls), 3)
        self.assertGreater(max_active, 1)


if __name__ == "__main__":
    unittest.main()
