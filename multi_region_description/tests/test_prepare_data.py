from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT_PATH)
assert SPEC and SPEC.loader
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


class PrepareDataTests(unittest.TestCase):
    def test_qwen_description_source_preserves_multi_region_loc_format(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.json"
            path.write_text(
                json.dumps(
                    {
                        "image": {"path": "/tmp/frame.jpg"},
                        "valid_crop_count": 2,
                        "crops": {
                            "person_002": {
                                "crop_index": 2,
                                "bbox_loc_0_999": [5, 6, 7, 8],
                                "florence": {"description": "Florence second"},
                                "qwen": {"status": "success", "description": "Qwen second"},
                            },
                            "person_001": {
                                "crop_index": 1,
                                "bbox_loc_0_999": [1, 2, 3, 4],
                                "florence": {"description": "Florence first"},
                                "qwen": {"status": "success", "description": "Qwen first"},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            sample, count = prepare_data.process_json(path, label_format="loc", description_source="qwen")

            self.assertEqual(count, 2)
            self.assertEqual(
                sample["prompt"],
                "<REGIONS_TO_DESCRIPTIONS><loc_1><loc_2><loc_3><loc_4><sep><loc_5><loc_6><loc_7><loc_8>",
            )
            self.assertEqual(
                sample["label"],
                "Qwen first<loc_1><loc_2><loc_3><loc_4>Qwen second<loc_5><loc_6><loc_7><loc_8>",
            )

    def test_florence_remains_default_description_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.json"
            path.write_text(
                json.dumps(
                    {
                        "image": {"path": "/tmp/frame.jpg"},
                        "valid_crop_count": 1,
                        "crops": {
                            "person_001": {
                                "crop_index": 1,
                                "bbox_loc_0_999": [1, 2, 3, 4],
                                "florence": {"description": "Florence text"},
                                "qwen": {"status": "success", "description": "Qwen text"},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            sample, count = prepare_data.process_json(path, label_format="sep")

            self.assertEqual(count, 1)
            self.assertEqual(sample["label"], "Florence text")


if __name__ == "__main__":
    unittest.main()
