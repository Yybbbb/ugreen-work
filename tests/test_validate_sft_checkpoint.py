import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validate_sft_checkpoint.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("validate_sft_checkpoint", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __len__(self):
        return 51289


class FakeProcessor:
    def __init__(self, prompt="What is the region <loc_1><loc_2><loc_3><loc_4>?"):
        self.prompt = prompt
        self.tokenizer = FakeTokenizer()
        self.tasks_answer_post_processing_type = {
            "<REGION_TO_CATEGORY>": "pure_text"
        }

    def _construct_prompts(self, prompts):
        return [self.prompt]


class ProcessorContractTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_contract_accepts_native_prompt_and_vocab(self):
        result = self.mod.validate_processor_contract(
            FakeProcessor(), expected_vocab_size=51289
        )
        self.assertEqual(
            result["expanded_prompt"],
            "What is the region <loc_1><loc_2><loc_3><loc_4>?",
        )
        self.assertEqual(result["post_processing"], "pure_text")
        self.assertEqual(result["tokenizer_size"], 51289)

    def test_contract_rejects_modified_prompt(self):
        with self.assertRaisesRegex(ValueError, "prompt contract"):
            self.mod.validate_processor_contract(
                FakeProcessor(prompt="Describe this person."),
                expected_vocab_size=51289,
            )


if __name__ == "__main__":
    unittest.main()
