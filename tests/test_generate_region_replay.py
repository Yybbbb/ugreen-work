import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_region_replay.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("generate_region_replay", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate(scene, frame, position, score, scale="small"):
    return {
        "sample_id": f"{scene}/{frame}.json#{position}",
        "source_json_relative_path": f"{scene}/{frame}.json",
        "crop_position": position,
        "crop_index": position + 1,
        "caption": "An adult person wears a jacket and carries a bag.",
        "scene": scene,
        "session": f"{scene}/session",
        "scale": scale,
        "selection_score": score,
    }


class ReplaySelectionTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.rows = [
            candidate("scene_a", "frame_1", 0, 0.8, "small"),
            candidate("scene_a", "frame_1", 1, 0.9, "large"),
            candidate("scene_a", "frame_2", 0, 0.7, "medium"),
            candidate("scene_a", "frame_3", 0, 0.6, "tiny"),
            candidate("scene_b", "frame_1", 0, 0.8, "small"),
            candidate("scene_b", "frame_2", 0, 0.7, "medium"),
            candidate("scene_b", "frame_3", 0, 0.6, "tiny"),
        ]

    def test_selection_hits_scene_quotas_without_repeated_frames(self):
        selected = self.mod.select_replay_candidates(
            self.rows, {"scene_a": 2, "scene_b": 2}, seed=7
        )

        self.assertEqual(
            Counter(row["scene"] for row in selected),
            {"scene_a": 2, "scene_b": 2},
        )
        self.assertEqual(
            len({row["source_json_relative_path"] for row in selected}), 4
        )

    def test_selection_is_deterministic(self):
        first = self.mod.select_replay_candidates(
            self.rows, {"scene_a": 2, "scene_b": 2}, seed=7
        )
        second = self.mod.select_replay_candidates(
            reversed(self.rows), {"scene_a": 2, "scene_b": 2}, seed=7
        )
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )

    def test_selection_rejects_insufficient_unique_frames(self):
        with self.assertRaisesRegex(ValueError, "unique frames"):
            self.mod.select_replay_candidates(
                self.rows, {"scene_a": 4, "scene_b": 2}, seed=7
            )


class ReplayResumeTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_generate_missing_skips_completed_rows(self):
        selection = [
            {"sample_id": "a", "prompt": "prompt-a"},
            {"sample_id": "b", "prompt": "prompt-b"},
            {"sample_id": "c", "prompt": "prompt-c"},
        ]
        completed = {"a": {"sample_id": "a", "label": "existing"}}
        calls = []
        appended = []

        def generate_one(row):
            calls.append(row["sample_id"])
            return f"description-{row['sample_id']}", f"raw-{row['sample_id']}"

        generated = self.mod.generate_missing(
            selection,
            completed,
            generate_one,
            appended.append,
            max_new_samples=1,
        )

        self.assertEqual(generated, 1)
        self.assertEqual(calls, ["b"])
        self.assertEqual(appended[0]["label"], "description-b")
        self.assertEqual(appended[0]["raw_generation"], "raw-b")

    def test_generate_missing_rejects_empty_label(self):
        with self.assertRaisesRegex(ValueError, "Empty replay label"):
            self.mod.generate_missing(
                [{"sample_id": "a", "prompt": "prompt"}],
                {},
                lambda row: (" ", "raw"),
                lambda row: None,
            )

    def test_validate_replay_label_accepts_matching_native_locations(self):
        label = "person in a black jacket<loc_1><loc_2><loc_3><loc_4>"
        self.assertEqual(
            self.mod.validate_replay_label(label, [1, 2, 3, 4]), label
        )

    def test_validate_replay_label_rejects_mismatched_locations(self):
        with self.assertRaisesRegex(ValueError, "location tokens"):
            self.mod.validate_replay_label(
                "person<loc_5><loc_6><loc_7><loc_8>", [1, 2, 3, 4]
            )

    def test_validate_replay_label_rejects_malformed_location(self):
        with self.assertRaisesRegex(ValueError, "Malformed"):
            self.mod.validate_replay_label("person<loc_bad>", [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
