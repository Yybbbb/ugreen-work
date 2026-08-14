import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_person_sft_data.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("prepare_person_sft_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class PrepareSplitTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "frame.jpg"
        self.image_path.write_bytes(b"image")
        self.relative_json = Path("scene/frame.json")
        self.split_root = self.root / "train"
        self.manifest = self.root / "manifests/train.jsonl"
        self.caption = "An adult male wears a blue jacket."
        document = {
            "image": {
                "path": str(self.image_path),
                "width": 100,
                "height": 100,
            },
            "crops": [
                {
                    "crop_index": 1,
                    "expanded_bbox_xyxy": [1, 1, 20, 20],
                    "caption": "An adult female wears a red coat.",
                    "attributes": {"gender": "female"},
                    "annotation": {"status": "success"},
                },
                {
                    "crop_index": 2,
                    "expanded_bbox_xyxy": [10, 20, 60, 80],
                    "caption": self.caption,
                    "attributes": {
                        "gender": "male",
                        "upper_garment": {"color": "blue"},
                    },
                    "annotation": {"status": "success"},
                },
            ],
        }
        write_json(self.split_root / self.relative_json, document)
        self.manifest_row = {
            "sample_id": "scene/frame.json#1",
            "source_json_relative_path": self.relative_json.as_posix(),
            "crop_position": 1,
            "crop_index": 2,
            "caption": self.caption,
            "scene": "scene",
            "session": "scene/session",
            "scale": "medium",
        }
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(
            json.dumps(self.manifest_row) + "\n", encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_prepare_split_maps_crop_and_recomputes_prompt(self):
        rows = list(self.mod.prepare_split(self.manifest, self.split_root, "train"))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["sample_id"], "scene/frame.json#1")
        self.assertEqual(row["bbox_loc_0_999"], [100, 200, 600, 800])
        self.assertEqual(
            row["prompt"],
            "<REGION_TO_CATEGORY><loc_100><loc_200><loc_600><loc_800>",
        )
        self.assertEqual(row["label"], self.caption)
        self.assertEqual(row["attributes"]["upper_garment"]["color"], "blue")
        self.assertEqual(row["image"], str(self.image_path))
        self.assertEqual(row["task"], "<REGION_TO_CATEGORY>")

    def test_prepare_split_rejects_caption_mismatch(self):
        row = dict(self.manifest_row)
        row["caption"] = "Different caption."
        self.manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Caption mismatch"):
            list(self.mod.prepare_split(self.manifest, self.split_root, "train"))

    def test_prepare_split_uses_crop_index_after_split_compaction(self):
        source_path = self.split_root / self.relative_json
        document = json.loads(source_path.read_text(encoding="utf-8"))
        document["crops"] = [document["crops"][1]]
        write_json(source_path, document)

        rows = list(self.mod.prepare_split(self.manifest, self.split_root, "train"))

        self.assertEqual(rows[0]["crop_position"], 1)
        self.assertEqual(rows[0]["crop_index"], 2)
        self.assertEqual(rows[0]["bbox_loc_0_999"], [100, 200, 600, 800])

    def test_prepare_split_rejects_sample_id_position_mismatch(self):
        row = dict(self.manifest_row)
        row["sample_id"] = "scene/frame.json#0"
        self.manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "sample_id"):
            list(self.mod.prepare_split(self.manifest, self.split_root, "train"))

    def test_reviewed_rl_expected_count_is_5981(self):
        self.assertEqual(self.mod.EXPECTED_FULL_COUNTS["rl"], 5981)

    def test_failed_multisplit_rebuild_preserves_active_prepared_files(self):
        dev_root = self.root / "dev"
        dev_manifest = self.root / "manifests/dev.jsonl"
        source_document = json.loads(
            (self.split_root / self.relative_json).read_text(encoding="utf-8")
        )
        write_json(dev_root / self.relative_json, source_document)
        invalid_dev_row = dict(self.manifest_row)
        invalid_dev_row["caption"] = "Caption that does not match the reviewed crop."
        dev_manifest.write_text(json.dumps(invalid_dev_row) + "\n", encoding="utf-8")

        output_dir = self.root / "prepared"
        output_dir.mkdir()
        old_train = b"old train prepared\n"
        old_dev = b"old dev prepared\n"
        old_metadata = b'{"version":"old"}\n'
        (output_dir / "train.jsonl").write_bytes(old_train)
        (output_dir / "dev.jsonl").write_bytes(old_dev)
        (output_dir / "metadata.json").write_bytes(old_metadata)

        with self.assertRaisesRegex(ValueError, "Caption mismatch"):
            self.mod.build_prepared_data(
                self.root,
                output_dir,
                ("train", "dev"),
                overwrite=True,
            )

        self.assertEqual((output_dir / "train.jsonl").read_bytes(), old_train)
        self.assertEqual((output_dir / "dev.jsonl").read_bytes(), old_dev)
        self.assertEqual((output_dir / "metadata.json").read_bytes(), old_metadata)

    def test_successful_rebuild_publishes_split_and_metadata(self):
        output_dir = self.root / "prepared"
        output_dir.mkdir()
        (output_dir / "train.jsonl").write_text("old\n", encoding="utf-8")
        (output_dir / "metadata.json").write_text("{}\n", encoding="utf-8")

        metadata = self.mod.build_prepared_data(
            self.root,
            output_dir,
            ("train",),
            overwrite=True,
        )

        prepared_rows = [
            json.loads(line)
            for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        published_metadata = json.loads(
            (output_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual([row["sample_id"] for row in prepared_rows], [self.manifest_row["sample_id"]])
        self.assertEqual(published_metadata, metadata)
        self.assertEqual(published_metadata["splits"]["train"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
