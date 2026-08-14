import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_person_splits.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("build_person_splits", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_session_key_handles_nested_and_flat_scenes(self):
        self.assertEqual(
            self.mod.session_key(Path("tradeshow/2025-01-01_10-00/frame_20.json")),
            "tradeshow/2025-01-01_10-00",
        )
        self.assertEqual(
            self.mod.session_key(Path("company/camera001_0042.json")),
            "company/camera001",
        )

    def test_exact_session_subset_is_deterministic(self):
        sizes = {"a": 9, "b": 22, "c": 28, "d": 14, "e": 8}

        first = self.mod.choose_sessions_exact(sizes, target=45, seed=7)
        second = self.mod.choose_sessions_exact(sizes, target=45, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(sum(sizes[key] for key in first), 45)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_ranked_selection_hits_person_and_frame_targets(self):
        records = []
        identifier = 0
        for frame in range(4):
            for crop in range(3):
                records.append(
                    self.mod.Sample.for_test(
                        identifier=identifier,
                        scene="scene",
                        session=f"session_{frame // 2}",
                        frame=f"frame_{frame}.json",
                        crop_position=crop,
                        score=100 - identifier,
                    )
                )
                identifier += 1

        selected = self.mod.select_ranked_frames(
            records,
            quota=6,
            frame_target=3,
            session_person_cap=4,
        )

        self.assertEqual(len(selected), 6)
        self.assertEqual(len({record.relative_json for record in selected}), 3)
        counts = {}
        for record in selected:
            counts[record.session] = counts.get(record.session, 0) + 1
        self.assertLessEqual(max(counts.values()), 4)

    def test_rl_quotas_sum_to_requested_total(self):
        quotas = self.mod.proportional_quotas(
            {"a": 7, "b": 8, "c": 15}, total=6
        )

        self.assertEqual(sum(quotas.values()), 6)
        self.assertEqual(quotas, {"a": 1, "b": 2, "c": 3})


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_materialize_filters_crops_and_preserves_nested_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            output_root = root / "output"
            relative = Path("tradeshow/session_a/frame_1.json")
            source = source_root / relative
            source.parent.mkdir(parents=True)
            payload = {
                "person_crop_count": 2,
                "crops": [
                    {"crop_index": 1, "caption": "first"},
                    {"crop_index": 2, "caption": "second"},
                ],
            }
            source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            source_bytes = source.read_bytes()
            record = self.mod.Sample.for_test(
                identifier=1,
                scene="tradeshow",
                session="tradeshow/session_a",
                frame=str(relative),
                crop_position=1,
                score=1.0,
            )

            frames, crops = self.mod.materialize_split(
                [record], source_root=source_root, output_root=output_root
            )

            written = json.loads((output_root / relative).read_text(encoding="utf-8"))
            self.assertEqual((frames, crops), (1, 1))
            self.assertEqual(written["person_crop_count"], 1)
            self.assertEqual(written["crops"][0]["crop_index"], 2)
            self.assertEqual(source.read_bytes(), source_bytes)


if __name__ == "__main__":
    unittest.main()
