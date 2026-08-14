import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "rl_service_config.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rl_service_config", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ServiceSpecTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_extractor_contract(self):
        spec = self.mod.EXTRACTOR_SERVICE
        self.assertEqual(spec.role, "extractor")
        self.assertEqual(spec.model, "Qwen3.6-27B-FP8")
        self.assertEqual(spec.model_path, Path("/data1/work/MichaelYu/models/Qwen3.6-27B-FP8"))
        self.assertEqual(spec.gpu, 1)
        self.assertEqual(spec.host_port, 6097)
        self.assertEqual(spec.base_url, "http://127.0.0.1:6097/v1")
        self.assertEqual(spec.image, "docker.m.daocloud.io/vllm/vllm-openai:v0.19.1")
        self.assertEqual(spec.vllm_version, "0.19.1")
        self.assertEqual(spec.max_model_len, 65536)
        self.assertEqual(spec.max_num_seqs, 64)
        self.assertEqual(spec.max_num_batched_tokens, 8192)
        self.assertEqual(spec.environment, (("VLLM_TEST_FORCE_FP8_MARLIN", "1"),))

    def test_judge_contract(self):
        spec = self.mod.JUDGE_SERVICE
        self.assertEqual(spec.role, "judge")
        self.assertEqual(spec.model, "Qwen3.5-4B")
        self.assertEqual(spec.model_path, Path("/data1/work/MichaelYu/models/Qwen3.5-4B"))
        self.assertEqual(spec.gpu, 2)
        self.assertEqual(spec.host_port, 6098)
        self.assertEqual(spec.base_url, "http://127.0.0.1:6098/v1")
        self.assertEqual(spec.max_model_len, 32768)
        self.assertEqual(spec.max_num_seqs, 128)
        self.assertEqual(spec.max_num_batched_tokens, 65536)
        self.assertEqual(spec.swap_space_gib, 8)
        self.assertEqual(spec.environment, ())

    def test_role_lookup(self):
        self.assertIs(self.mod.service_for_role("extractor"), self.mod.EXTRACTOR_SERVICE)
        self.assertIs(self.mod.service_for_role("judge"), self.mod.JUDGE_SERVICE)
        with self.assertRaisesRegex(ValueError, "unknown reward service role"):
            self.mod.service_for_role("other")


class DockerCommandTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_extractor_command_has_marlin_and_capacity_flags(self):
        argv = self.mod.docker_run_argv(self.mod.EXTRACTOR_SERVICE)
        self.assertEqual(argv[:3], ["docker", "run", "--detach"])
        self.assertIn("vllm-qwen36-27b-fp8", argv)
        self.assertIn("device=1", argv)
        self.assertIn("6097:8000", argv)
        self.assertIn("/data1/work/MichaelYu/models/Qwen3.6-27B-FP8:/models:ro", argv)
        self.assertIn("VLLM_TEST_FORCE_FP8_MARLIN=1", argv)
        self.assertIn("--language-model-only", argv)
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "65536")
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "64")
        self.assertEqual(argv[argv.index("--max-num-batched-tokens") + 1], "8192")
        self.assertEqual(
            argv[argv.index("--default-chat-template-kwargs") + 1],
            '{"enable_thinking": false}',
        )

    def test_judge_command_has_expected_capacity(self):
        argv = self.mod.docker_run_argv(self.mod.JUDGE_SERVICE)
        self.assertIn("device=2", argv)
        self.assertIn("6098:8000", argv)
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "32768")
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "128")
        self.assertEqual(argv[argv.index("--max-num-batched-tokens") + 1], "65536")
        self.assertEqual(argv[argv.index("--swap-space") + 1], "8")

    def test_command_has_no_destructive_or_auto_remove_flags(self):
        for spec in self.mod.SERVICES:
            argv = self.mod.docker_run_argv(spec)
            self.assertNotIn("--rm", argv)
            self.assertNotIn("rm", argv)
            self.assertNotIn("stop", argv)

    def test_metadata_is_json_serializable_shape(self):
        metadata = self.mod.service_metadata(self.mod.EXTRACTOR_SERVICE)
        self.assertEqual(metadata["model"], "Qwen3.6-27B-FP8")
        self.assertEqual(metadata["gpu"], 1)
        self.assertEqual(metadata["max_num_batched_tokens"], 8192)
        self.assertEqual(metadata["environment"], {"VLLM_TEST_FORCE_FP8_MARLIN": "1"})


if __name__ == "__main__":
    unittest.main()
