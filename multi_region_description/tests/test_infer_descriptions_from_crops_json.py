from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/infer_descriptions_from_crops_json.py"
)
SPEC = importlib.util.spec_from_file_location("infer_crops_json", SCRIPT_PATH)
assert SPEC and SPEC.loader
infer_crops_json = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = infer_crops_json
SPEC.loader.exec_module(infer_crops_json)


class InferDescriptionsFromCropsJsonTests(unittest.TestCase):
    def test_single_crop_uses_region_to_description(self) -> None:
        crops = [infer_crops_json.CropInput("person_001", 1, (1, 2, 3, 4))]

        task, task_prompt, expanded_prompt = infer_crops_json.build_prompts(crops)

        self.assertEqual(task, "<REGION_TO_DESCRIPTION>")
        self.assertEqual(
            task_prompt,
            "<REGION_TO_DESCRIPTION><loc_1><loc_2><loc_3><loc_4>",
        )
        self.assertEqual(
            expanded_prompt,
            "What does the region <loc_1><loc_2><loc_3><loc_4> describe?",
        )

    def test_multiple_crops_are_sorted_and_use_multi_region_task(self) -> None:
        payload = {
            "image": {"path": "/tmp/image.jpg", "width": 100, "height": 200},
            "crops": {
                "person_002": {
                    "crop_index": 2,
                    "bbox_loc_0_999": [5, 6, 7, 8],
                },
                "person_001": {
                    "crop_index": 1,
                    "bbox_loc_0_999": [1, 2, 3, 4],
                },
            },
        }

        crops = infer_crops_json.extract_crops(payload)
        task, task_prompt, expanded_prompt = infer_crops_json.build_prompts(crops)

        self.assertEqual([crop.crop_key for crop in crops], ["person_001", "person_002"])
        self.assertEqual(task, "<REGIONS_TO_DESCRIPTIONS>")
        self.assertEqual(
            task_prompt,
            "<REGIONS_TO_DESCRIPTIONS><loc_1><loc_2><loc_3><loc_4>"
            "<sep><loc_5><loc_6><loc_7><loc_8>",
        )
        self.assertEqual(
            expanded_prompt,
            "What does each region <loc_1><loc_2><loc_3><loc_4>"
            "<sep><loc_5><loc_6><loc_7><loc_8> describe?",
        )

    def test_xyxy_bbox_is_normalized_when_loc_bbox_is_missing(self) -> None:
        payload = {
            "image": {"path": "/tmp/image.jpg", "width": 100, "height": 200},
            "crops": [{"crop_key": "person", "bbox_xyxy": [10, 20, 90, 180]}],
        }

        crops = infer_crops_json.extract_crops(payload)

        self.assertEqual(crops[0].bbox_loc_0_999, (100, 100, 899, 899))

    def test_multi_region_output_is_parsed_and_matched_by_bbox(self) -> None:
        raw = (
            "</s><s>A person in white<loc_1><loc_2><loc_3><loc_4>"
            "A person in black<loc_5><loc_6><loc_7><loc_8></s>"
        )
        crops = [
            infer_crops_json.CropInput("white", 1, (1, 2, 3, 4)),
            infer_crops_json.CropInput("black", 2, (5, 6, 7, 8)),
        ]

        predictions = infer_crops_json.parse_multi_region_output(raw)
        results = infer_crops_json.assign_multi_predictions(crops, predictions)

        self.assertEqual([item["description"] for item in results], ["A person in white", "A person in black"])
        self.assertEqual([item["match_method"] for item in results], ["bbox_exact", "bbox_exact"])

    def test_single_crop_selects_the_description_with_the_input_bbox(self) -> None:
        raw = (
            "A target person<loc_1><loc_2><loc_3><loc_4>"
            "An extra person<loc_5><loc_6><loc_7><loc_8>"
        )
        crop = infer_crops_json.CropInput("target", 1, (1, 2, 3, 4))

        result, predictions = infer_crops_json.assign_single_prediction(crop, raw)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(result["description"], "A target person")
        self.assertEqual(result["match_method"], "bbox_exact")


if __name__ == "__main__":
    unittest.main()
