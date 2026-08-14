import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "run_rl_reward_services.py"


def load_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("run_rl_reward_services", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SelectionAndPrintingTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_selects_roles_in_stable_order(self):
        self.assertEqual(
            [spec.role for spec in self.mod.selected_services("all")],
            ["extractor", "judge"],
        )
        self.assertEqual(
            [spec.role for spec in self.mod.selected_services("judge")],
            ["judge"],
        )

    def test_printed_command_is_shell_escaped_but_not_executed(self):
        command = self.mod.print_command(self.mod.EXTRACTOR_SERVICE)
        self.assertIn("docker run --detach", command)
        self.assertIn("VLLM_TEST_FORCE_FP8_MARLIN=1", command)
        self.assertNotIn("docker rm", command)
        self.assertNotIn("docker stop", command)


class ServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_existing_healthy_container_is_reused_without_run(self):
        run_calls = []
        result = self.mod.start_service(
            self.mod.EXTRACTOR_SERVICE,
            container_state_fn=lambda spec: "running",
            run_fn=lambda *args, **kwargs: run_calls.append((args, kwargs)),
            preflight_fn=lambda base_url, model: {"model": model, "content": "OK"},
        )
        self.assertEqual(result["action"], "reused")
        self.assertEqual(run_calls, [])

    def test_existing_unhealthy_container_is_not_replaced(self):
        run_calls = []

        def failed_preflight(base_url, model):
            raise RuntimeError("wrong model")

        with self.assertRaisesRegex(RuntimeError, "will not be replaced"):
            self.mod.start_service(
                self.mod.EXTRACTOR_SERVICE,
                container_state_fn=lambda spec: "running",
                run_fn=lambda *args, **kwargs: run_calls.append((args, kwargs)),
                preflight_fn=failed_preflight,
            )
        self.assertEqual(run_calls, [])

    def test_missing_container_runs_generated_argv_and_waits_until_ready(self):
        states = iter([None, "running"])
        run_calls = []

        def fake_run(argv, **kwargs):
            run_calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, stdout="container-id\n", stderr="")

        result = self.mod.start_service(
            self.mod.JUDGE_SERVICE,
            container_state_fn=lambda spec: next(states),
            run_fn=fake_run,
            preflight_fn=lambda base_url, model: {"model": model, "content": "OK"},
            timeout=1.0,
            sleep_fn=lambda seconds: None,
            monotonic_fn=iter([0.0, 0.1]).__next__,
        )
        self.assertEqual(result["action"], "started")
        self.assertEqual(run_calls[0][0], self.mod.docker_run_argv(self.mod.JUDGE_SERVICE))

    def test_status_propagates_unhealthy_service(self):
        with self.assertRaisesRegex(RuntimeError, "container state is exited"):
            self.mod.check_service(
                self.mod.JUDGE_SERVICE,
                container_state_fn=lambda spec: "exited",
                preflight_fn=lambda base_url, model: {"model": model},
            )


if __name__ == "__main__":
    unittest.main()
