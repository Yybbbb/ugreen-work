import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "infer_person_attribute.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("infer_person_attribute", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class InferenceManifestTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_rank_indices_are_disjoint_and_cover_rows(self):
        shards = [
            list(self.mod.rank_indices(11, rank, 4)) for rank in range(4)
        ]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(11)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_clean_prediction_removes_generation_padding_tokens(self):
        self.assertEqual(
            self.mod.clean_prediction(
                "An adult person wears a black jacket.<pad><pad>"
            ),
            "An adult person wears a black jacket.",
        )

    def test_completed_rows_are_loaded_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank.jsonl"
            path.write_text(
                json.dumps({"sample_id": "a", "prediction": "one"})
                + "\n"
                + json.dumps({"sample_id": "b", "prediction": "two"})
                + "\n",
                encoding="utf-8",
            )
            completed = self.mod.load_completed_predictions(path)
            self.assertEqual(sorted(completed), ["a", "b"])

            path.write_text(
                json.dumps({"sample_id": "a", "prediction": "one"})
                + "\n"
                + json.dumps({"sample_id": "a", "prediction": "duplicate"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                self.mod.load_completed_predictions(path)

    def test_merge_requires_exact_test_ids_and_preserves_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_rows = root / "test.jsonl"
            test_rows.write_text(
                "\n".join(
                    json.dumps({"sample_id": sample_id, "label": sample_id})
                    for sample_id in ("b", "a", "c")
                )
                + "\n",
                encoding="utf-8",
            )
            rank0 = root / "rank0.jsonl"
            rank1 = root / "rank1.jsonl"
            rank0.write_text(
                json.dumps({"sample_id": "a", "prediction": "A"}) + "\n"
                + json.dumps({"sample_id": "c", "prediction": "C"}) + "\n",
                encoding="utf-8",
            )
            rank1.write_text(
                json.dumps({"sample_id": "b", "prediction": "B"}) + "\n",
                encoding="utf-8",
            )
            output = root / "merged.jsonl"
            count = self.mod.merge_predictions(test_rows, [rank0, rank1], output)
            self.assertEqual(count, 3)
            merged = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["sample_id"] for row in merged], ["b", "a", "c"])
            self.assertEqual([row["prediction"] for row in merged], ["B", "A", "C"])

            limited = root / "limited.jsonl"
            self.assertEqual(
                self.mod.merge_predictions(
                    test_rows, [rank0, rank1], limited, max_rows=2
                ),
                2,
            )
            limited_rows = [
                json.loads(line) for line in limited.read_text().splitlines()
            ]
            self.assertEqual([row["sample_id"] for row in limited_rows], ["b", "a"])

            rank1.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing sample_id"):
                self.mod.merge_predictions(test_rows, [rank0, rank1], root / "bad.jsonl")


if __name__ == "__main__":
    unittest.main()
