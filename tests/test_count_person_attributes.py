import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from count_person_attributes import PersonAttributeParser


class PersonAttributeParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = PersonAttributeParser()

    def parse(self, description: str) -> dict[str, object]:
        return self.parser.parse(description)

    def categories(self, description: str) -> dict[str, list[str]]:
        result = self.parse(description)
        return {
            item["category"]: item["values"]
            for item in result["attributes"]
        }

    def test_black_dress_is_color_and_type(self) -> None:
        result = self.parse("A person in a black dress")
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            self.categories("A person in a black dress"),
            {"clothing_type": ["dress"], "clothing_color": ["black"]},
        )

    def test_nonstandard_color_and_type_are_preserved(self) -> None:
        result = self.parse("A person wearing a beige coat")
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            self.categories("A person wearing a beige coat"),
            {"upper_type": ["coat"], "upper_color": ["beige"]},
        )

    def test_generic_outfit_is_open_clothing_type(self) -> None:
        result = self.parse("A person in a dark outfit")
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            self.categories("A person in a dark outfit"),
            {"clothing_type": ["outfit"], "clothing_color": ["dark"]},
        )

    def test_gender_age_clothing_and_open_hand_object(self) -> None:
        description = "A teenage woman in a beige coat holds a red laptop"
        result = self.parse(description)
        self.assertEqual(result["count"], 6)
        self.assertEqual(
            self.categories(description),
            {
                "gender": ["female"],
                "age": ["teenager"],
                "upper_type": ["coat"],
                "upper_color": ["beige"],
                "hand_object": ["laptop"],
                "hand_object_color": ["red"],
            },
        )

    def test_boy_implies_gender_and_child_age(self) -> None:
        result = self.parse("A boy stands in a room")
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            self.categories("A boy stands in a room"),
            {"gender": ["male"], "age": ["child"]},
        )

    def test_hair_color_and_length_are_one_attribute(self) -> None:
        description = "A woman with long black hair sits at a desk"
        result = self.parse(description)
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            self.categories(description),
            {"gender": ["female"], "hair": ["long black"]},
        )

    def test_multiple_hair_details_still_count_once(self) -> None:
        description = "A young woman with long black hair tied back in a ponytail"
        result = self.parse(description)
        self.assertEqual(result["count"], 3)
        self.assertEqual(
            self.categories(description),
            {
                "gender": ["female"],
                "age": ["young"],
                "hair": ["long black", "ponytail", "tied back"],
            },
        )

    def test_bald_is_hair_attribute(self) -> None:
        description = "A bald man wearing a white shirt"
        result = self.parse(description)
        self.assertEqual(result["count"], 4)
        self.assertEqual(
            self.categories(description),
            {
                "gender": ["male"],
                "hair": ["bald"],
                "upper_type": ["shirt"],
                "upper_color": ["white"],
            },
        )

    def test_shorts_imply_short_lower_length(self) -> None:
        result = self.parse("A person wearing black shorts")
        self.assertEqual(result["count"], 3)
        self.assertEqual(
            self.categories("A person wearing black shorts"),
            {
                "lower_type": ["shorts"],
                "lower_color": ["black"],
                "lower_length": ["short"],
            },
        )

    def test_t_shirt_does_not_imply_sleeve_length(self) -> None:
        result = self.parse("A person wearing a white t-shirt")
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            self.categories("A person wearing a white t-shirt"),
            {"upper_type": ["t-shirt"], "upper_color": ["white"]},
        )

    def test_explicit_sleeve_length_is_counted(self) -> None:
        result = self.parse("A person wearing a white long-sleeve shirt")
        self.assertEqual(result["count"], 3)
        self.assertEqual(
            self.categories("A person wearing a white long-sleeve shirt"),
            {
                "upper_type": ["shirt"],
                "upper_color": ["white"],
                "upper_sleeve_length": ["long"],
            },
        )

    def test_background_colors_are_ignored(self) -> None:
        description = "A man in a black shirt stands near a white couch and a red wall"
        result = self.parse(description)
        self.assertEqual(result["count"], 3)
        self.assertEqual(
            self.categories(description),
            {
                "gender": ["male"],
                "upper_type": ["shirt"],
                "upper_color": ["black"],
            },
        )

    def test_holding_backpack_is_hand_object(self) -> None:
        description = "A person holding a white backpack"
        result = self.parse(description)
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            self.categories(description),
            {"hand_object": ["backpack"], "hand_object_color": ["white"]},
        )

    def test_wearing_backpack_is_back_object(self) -> None:
        description = "A person wearing a black backpack"
        result = self.parse(description)
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            self.categories(description),
            {"back_object": ["backpack"], "back_object_color": ["black"]},
        )

    def test_carrying_backpack_defaults_to_back_object(self) -> None:
        description = "A person carrying a blue backpack"
        result = self.parse(description)
        self.assertEqual(result["count"], 2)
        self.assertEqual(
            self.categories(description),
            {"back_object": ["backpack"], "back_object_color": ["blue"]},
        )

    def test_held_jacket_is_not_worn_clothing(self) -> None:
        description = "A person wearing a white shirt while holding a black jacket"
        result = self.parse(description)
        self.assertEqual(result["count"], 4)
        self.assertEqual(
            self.categories(description),
            {
                "upper_type": ["shirt"],
                "upper_color": ["white"],
                "hand_object": ["jacket"],
                "hand_object_color": ["black"],
            },
        )

    def test_glasses_and_mask_are_counted_once_each(self) -> None:
        description = "A woman wearing glasses and a white face mask"
        result = self.parse(description)
        self.assertEqual(result["count"], 3)
        self.assertEqual(
            self.categories(description),
            {"gender": ["female"], "glasses": ["with"], "face_covering": ["with"]},
        )

    def test_object_held_in_arms_is_embraced_not_hand_object(self) -> None:
        description = "A woman holding a white box in her arms"
        result = self.parse(description)
        self.assertEqual(result["count"], 3)
        self.assertEqual(
            self.categories(description),
            {
                "gender": ["female"],
                "embraced_object": ["box"],
                "embraced_object_color": ["white"],
            },
        )

    def test_multicolor_garment_color_counts_once(self) -> None:
        description = "A woman in a black and white checkered dress"
        result = self.parse(description)
        self.assertEqual(result["count"], 3)
        self.assertEqual(
            self.categories(description),
            {
                "gender": ["female"],
                "clothing_type": ["dress"],
                "clothing_color": ["black and white"],
            },
        )


if __name__ == "__main__":
    unittest.main()
