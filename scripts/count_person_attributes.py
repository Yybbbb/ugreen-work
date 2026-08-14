#!/usr/bin/env python3
"""Count person attributes explicitly expressed in an English description."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field


COLOR_WORDS = (
    "black",
    "white",
    "gray",
    "grey",
    "red",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "orange",
    "brown",
    "beige",
    "navy",
    "teal",
    "cyan",
    "maroon",
    "khaki",
    "gold",
    "golden",
    "silver",
    "cream",
    "tan",
    "dark",
    "light",
    "light-colored",
    "dark-colored",
)

COLOR_ATOM = "(?:" + "|".join(re.escape(value) for value in COLOR_WORDS) + ")"
COLOR_EXPRESSION = rf"{COLOR_ATOM}(?:\s+{COLOR_ATOM})?(?:\s+and\s+{COLOR_ATOM}(?:\s+{COLOR_ATOM})?)?"

UPPER_TYPES = (
    "reflective vest",
    "tank top",
    "t-shirt",
    "tee shirt",
    "sweatshirt",
    "camisole",
    "cardigan",
    "windbreaker",
    "raincoat",
    "blazer",
    "jacket",
    "hoodie",
    "sweater",
    "shirt",
    "blouse",
    "coat",
    "vest",
    "jersey",
    "polo",
    "top",
)
LOWER_TYPES = (
    "wide-leg pants",
    "cargo pants",
    "sweatpants",
    "trousers",
    "leggings",
    "shorts",
    "jeans",
    "pants",
    "skirt",
)
ONE_PIECE_TYPES = (
    "jumpsuit",
    "overalls",
    "clothing",
    "uniform",
    "clothes",
    "outfit",
    "dress",
    "suit",
    "robe",
)
SHOE_TYPES = ("sneakers", "sandals", "slippers", "boots", "heels", "shoes")
HAT_TYPES = ("baseball cap", "beanie", "helmet", "hat", "cap")
BACK_OBJECT_TYPES = (
    "crossbody bag",
    "messenger bag",
    "shoulder bag",
    "back pack",
    "backpack",
)

TYPE_ALIASES = {
    "tee shirt": "t-shirt",
    "back pack": "backpack",
    "grey": "gray",
}

PERSON_WORD = r"(?:person|people|man|men|woman|women|boy|boys|girl|girls|male|female|adult|child|teenager)"

GENDER_PATTERNS = (
    ("female", re.compile(r"\b(?:woman|women|female|lady|ladies|girl|girls)\b")),
    ("male", re.compile(r"\b(?:man|men|male|gentleman|gentlemen|boy|boys)\b")),
)
AGE_PATTERNS = (
    ("baby", re.compile(r"\b(?:baby|babies|infant|infants|toddler|toddlers)\b")),
    ("child", re.compile(r"\b(?:child|children|kid|kids|boy|boys|girl|girls)\b")),
    ("teenager", re.compile(r"\b(?:teenager|teenagers|teenage|teenaged|teen)\b")),
    ("middle-aged", re.compile(r"\bmiddle[ -]aged\b")),
    ("elderly", re.compile(rf"\b(?:elderly|senior|seniors|older|aged)\b|\bold\s+{PERSON_WORD}\b")),
    ("young", re.compile(r"\byoung\s+(?:person|people|man|men|woman|women|male|female|boy|girl|adult)\b")),
    ("adult", re.compile(r"\badult(?:s)?\b")),
)

HAIR_DESCRIPTOR = (
    r"(?:very\s+)?(?:long|short|medium(?:[ -]length)?|shoulder[ -]length|"
    r"black|white|gray|grey|red|ginger|blond|blonde|brown|dark|light|"
    r"curly|straight|wavy|coily|thick|thin|shaved)"
)
HAIR_PATTERN = re.compile(
    rf"\b(?P<value>{HAIR_DESCRIPTOR}(?:\s+{HAIR_DESCRIPTOR}){{0,3}})\s+hair\b",
    re.IGNORECASE,
)
HAIR_STYLE_PATTERN = re.compile(
    r"\b(?P<value>ponytail|pigtails?|braids?|dreadlocks?|cornrows?|afro|bun|bald|balding)\b",
    re.IGNORECASE,
)
HAIR_TIED_PATTERN = re.compile(
    r"\bhair\s+(?:is\s+)?(?P<value>tied back|pulled back|in a ponytail|in a bun)\b",
    re.IGNORECASE,
)

HELD_ACTION = re.compile(r"\b(?:holding|holds?|carrying|carries|carry)\b", re.IGNORECASE)
HAND_ACTION_PATTERN = re.compile(
    r"\b(?P<action>holding|holds?|carrying|carries|carry|using|uses|looking at|looks at)\s+"
    r"(?P<object>.+?)"
    r"(?=\s+(?:while|near|beside|inside|outside|during|who|that|where|"
    r"stands?|sits?|walks?|runs?|rides?|bends?|lies|is|are)\b|[,.;]|$)",
    re.IGNORECASE,
)
EMBRACED_PATTERN = re.compile(
    r"\b(?P<action>hugging|hugs?|cradling|cradles?|embracing|embraces?)\s+"
    r"(?P<object>.+?)"
    r"(?=\s+(?:while|near|beside|inside|outside|stands?|sits?|walks?|is|are)\b|[,.;]|$)",
    re.IGNORECASE,
)
IN_ARMS_PATTERN = re.compile(
    r"\b(?:holding|holds?|carrying|carries)\s+(?P<object>.+?)\s+"
    r"(?:in|against)\s+(?:his|her|their|the)\s+(?:arms?|chest)\b",
    re.IGNORECASE,
)

NEGATED_GLASSES = re.compile(r"\b(?:without|no|not wearing)\s+(?:any\s+)?(?:sun)?glasses\b")
POSITIVE_GLASSES = re.compile(r"\b(?:wearing\s+|with\s+)?(?:sun)?glasses\b")
NEGATED_MASK = re.compile(r"\b(?:without|no|not wearing)\s+(?:a\s+)?(?:face\s+)?mask\b")
POSITIVE_MASK = re.compile(r"\b(?:wearing\s+|with\s+)?(?:a\s+)?(?:face\s+)?mask\b")

LOC_TOKEN = re.compile(r"<loc_\d+>|<sep>|<s>|</s>|<pad>", re.IGNORECASE)
SPACE = re.compile(r"\s+")


@dataclass
class AttributeValue:
    values: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def add(self, value: str, evidence: str) -> None:
        value = TYPE_ALIASES.get(value.strip().lower(), value.strip().lower())
        evidence = SPACE.sub(" ", evidence.strip())
        if value and value not in self.values:
            self.values.append(value)
        if evidence and evidence not in self.evidence:
            self.evidence.append(evidence)


class PersonAttributeParser:
    """Rule-based parser that counts unique person-attribute categories."""

    def __init__(self) -> None:
        self.type_patterns = {
            "upper": self._compile_type_pattern(UPPER_TYPES),
            "lower": self._compile_type_pattern(LOWER_TYPES),
            "one_piece": self._compile_type_pattern(ONE_PIECE_TYPES),
            "shoe": self._compile_type_pattern(SHOE_TYPES),
            "hat": self._compile_type_pattern(HAT_TYPES),
        }

    @staticmethod
    def _compile_type_pattern(values: tuple[str, ...]) -> re.Pattern[str]:
        alternatives = "|".join(
            re.escape(value).replace(r"\ ", r"[ -]")
            for value in sorted(values, key=len, reverse=True)
        )
        return re.compile(rf"(?<![\w-])(?P<type>{alternatives})(?![\w-])", re.IGNORECASE)

    @staticmethod
    def _normalize(description: str) -> str:
        description = LOC_TOKEN.sub(" ", description)
        description = description.replace("–", "-").replace("—", "-")
        return SPACE.sub(" ", description).strip().lower()

    @staticmethod
    def _add(
        attributes: dict[str, AttributeValue],
        category: str,
        value: str,
        evidence: str,
    ) -> None:
        attributes.setdefault(category, AttributeValue()).add(value, evidence)

    @staticmethod
    def _is_held_mention(text: str, start: int) -> bool:
        prefix = text[max(0, start - 70):start]
        last_boundary = max(prefix.rfind(","), prefix.rfind("."), prefix.rfind(";"))
        prefix = prefix[last_boundary + 1:]
        match = list(HELD_ACTION.finditer(prefix))
        if not match:
            return False
        tail = prefix[match[-1].end():]
        return len(tail.split()) <= 7 and not re.search(r"\bwearing\b", tail)

    @staticmethod
    def _color_before(text: str, start: int) -> tuple[str, str] | None:
        prefix = text[max(0, start - 55):start]
        match = re.search(
            rf"(?P<color>{COLOR_EXPRESSION})"
            rf"(?:\s+(?:plaid|striped|checkered|patterned|colored|(?:long|short)[ -]sleeve(?:d)?))*\s*$",
            prefix,
            re.IGNORECASE,
        )
        if not match:
            return None
        color = SPACE.sub(" ", match.group("color").lower())
        evidence = text[max(0, start - len(match.group(0))):start]
        return color, evidence.strip()

    def _parse_demographics(self, text: str, attributes: dict[str, AttributeValue]) -> None:
        for value, pattern in GENDER_PATTERNS:
            match = pattern.search(text)
            if match:
                self._add(attributes, "gender", value, match.group(0))
        for value, pattern in AGE_PATTERNS:
            match = pattern.search(text)
            if match:
                self._add(attributes, "age", value, match.group(0))

    def _parse_hair(self, text: str, attributes: dict[str, AttributeValue]) -> None:
        for pattern in (HAIR_PATTERN, HAIR_STYLE_PATTERN, HAIR_TIED_PATTERN):
            for match in pattern.finditer(text):
                value = match.group("value").lower().replace("grey", "gray")
                self._add(attributes, "hair", value, match.group(0))

    def _parse_clothing(self, text: str, attributes: dict[str, AttributeValue]) -> None:
        category_names = {
            "upper": ("upper_type", "upper_color"),
            "lower": ("lower_type", "lower_color"),
            "one_piece": ("clothing_type", "clothing_color"),
        }
        for group, (type_category, color_category) in category_names.items():
            for match in self.type_patterns[group].finditer(text):
                if self._is_held_mention(text, match.start()):
                    continue
                garment_type = match.group("type").lower()
                garment_type = TYPE_ALIASES.get(garment_type, garment_type)
                self._add(attributes, type_category, garment_type, match.group(0))
                color_match = self._color_before(text, match.start())
                if color_match:
                    color, prefix = color_match
                    self._add(
                        attributes,
                        color_category,
                        color,
                        f"{prefix} {match.group(0)}",
                    )

        for match in self.type_patterns["upper"].finditer(text):
            if self._is_held_mention(text, match.start()):
                continue
            prefix = text[max(0, match.start() - 30):match.start()]
            sleeve = re.search(r"\b(?P<length>long|short)[ -]sleeve(?:d)?\s*$", prefix)
            if sleeve:
                self._add(
                    attributes,
                    "upper_sleeve_length",
                    sleeve.group("length"),
                    f"{sleeve.group(0)} {match.group(0)}",
                )

        for match in self.type_patterns["lower"].finditer(text):
            if self._is_held_mention(text, match.start()):
                continue
            lower_type = match.group("type").lower()
            if lower_type == "shorts":
                self._add(attributes, "lower_length", "short", match.group(0))
                continue
            prefix = text[max(0, match.start() - 18):match.start()]
            length = re.search(r"\b(?P<length>long|short)\s*$", prefix)
            if length:
                self._add(
                    attributes,
                    "lower_length",
                    length.group("length"),
                    f"{length.group(0)} {match.group(0)}",
                )

        for group, category in (("shoe", "shoe_color"), ("hat", "hat_color")):
            for match in self.type_patterns[group].finditer(text):
                if self._is_held_mention(text, match.start()):
                    continue
                color_match = self._color_before(text, match.start())
                if color_match:
                    color, prefix = color_match
                    self._add(attributes, category, color, f"{prefix} {match.group(0)}")

    @staticmethod
    def _clean_object_phrase(phrase: str) -> str:
        phrase = re.sub(r"^(?:a|an|the|his|her|their|its|some)\s+", "", phrase.strip())
        phrase = re.sub(r"\s+(?:in|on|at|with)\s+(?:his|her|their|the|a|an)\b.*$", "", phrase)
        return SPACE.sub(" ", phrase).strip()

    @staticmethod
    def _object_color(phrase: str) -> str | None:
        match = re.search(rf"\b(?P<color>{COLOR_EXPRESSION})\b", phrase, re.IGNORECASE)
        return SPACE.sub(" ", match.group("color").lower()) if match else None

    @staticmethod
    def _object_type(phrase: str) -> str:
        value = phrase.lower()
        value = re.sub(rf"^(?:large|small|little|big|tiny|cardboard|plastic|paper|shopping)\s+", "", value)
        value = re.sub(rf"^(?:{COLOR_EXPRESSION})\s+", "", value)
        value = re.sub(r"^(?:large|small|little|big|tiny)\s+", "", value)
        return SPACE.sub(" ", value).strip()

    def _add_object(
        self,
        attributes: dict[str, AttributeValue],
        category: str,
        color_category: str,
        phrase: str,
        evidence: str,
    ) -> None:
        phrase = self._clean_object_phrase(phrase)
        if not phrase:
            return
        object_type = self._object_type(phrase)
        if object_type:
            self._add(attributes, category, object_type, evidence)
        color = self._object_color(phrase)
        if color:
            self._add(attributes, color_category, color, evidence)

    def _parse_objects(self, text: str, attributes: dict[str, AttributeValue]) -> None:
        embraced_spans: list[tuple[int, int]] = []
        for pattern in (EMBRACED_PATTERN, IN_ARMS_PATTERN):
            for match in pattern.finditer(text):
                embraced_spans.append(match.span())
                self._add_object(
                    attributes,
                    "embraced_object",
                    "embraced_object_color",
                    match.group("object"),
                    match.group(0),
                )

        for match in HAND_ACTION_PATTERN.finditer(text):
            if any(start <= match.start() < end for start, end in embraced_spans):
                continue
            phrase = match.group("object")
            if re.fullmatch(r"(?:his|her|their)\s+(?:hands?|arms?)", phrase.strip()):
                continue
            carries_back_object = re.search(
                r"\b(?:backpack|back pack|shoulder bag|messenger bag|crossbody bag)\b",
                phrase,
            )
            explicitly_in_hand = re.search(r"\b(?:in|by)\s+(?:his|her|their|the)\s+hand\b", phrase)
            if (
                match.group("action").lower().startswith("carr")
                and carries_back_object
                and not explicitly_in_hand
            ):
                continue
            self._add_object(
                attributes,
                "hand_object",
                "hand_object_color",
                phrase,
                match.group(0),
            )

        back_pattern = self._compile_type_pattern(BACK_OBJECT_TYPES)
        for match in back_pattern.finditer(text):
            prefix = text[max(0, match.start() - 55):match.start()]
            suffix = text[match.end():match.end() + 45]
            if re.search(r"\b(?:holding|holds?)\b[^,.;]{0,35}$", prefix):
                continue
            if re.search(r"\b(?:in|by)\s+(?:his|her|their|the)\s+hand\b", suffix):
                continue
            if not re.search(r"\b(?:wearing|carrying|carries|with|has|having)\b[^,.;]{0,45}$", prefix):
                if not re.search(r"\b(?:on|over)\s+(?:his|her|their|the)\s+(?:back|shoulders?)\b", suffix):
                    continue
            color_match = self._color_before(text, match.start())
            self._add(attributes, "back_object", match.group("type"), match.group(0))
            if color_match:
                color, color_evidence = color_match
                self._add(
                    attributes,
                    "back_object_color",
                    color,
                    f"{color_evidence} {match.group(0)}",
                )

    def _parse_face_accessories(self, text: str, attributes: dict[str, AttributeValue]) -> None:
        match = NEGATED_GLASSES.search(text)
        if match:
            self._add(attributes, "glasses", "none", match.group(0))
        else:
            match = POSITIVE_GLASSES.search(text)
            if match and not self._is_held_mention(text, match.start()):
                self._add(attributes, "glasses", "with", match.group(0))

        match = NEGATED_MASK.search(text)
        if match:
            self._add(attributes, "face_covering", "none", match.group(0))
        else:
            match = POSITIVE_MASK.search(text)
            if match and not self._is_held_mention(text, match.start()):
                self._add(attributes, "face_covering", "with", match.group(0))

    def parse(self, description: str) -> dict[str, object]:
        text = self._normalize(description)
        attributes: dict[str, AttributeValue] = {}
        self._parse_demographics(text, attributes)
        self._parse_hair(text, attributes)
        self._parse_clothing(text, attributes)
        self._parse_objects(text, attributes)
        self._parse_face_accessories(text, attributes)
        return {
            "description": description,
            "count": len(attributes),
            "attributes": [
                {
                    "category": category,
                    "values": value.values,
                    "evidence": value.evidence,
                }
                for category, value in attributes.items()
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count person attributes expressed in one English description."
    )
    parser.add_argument(
        "description",
        nargs="?",
        help="Description text. If omitted, non-empty lines are read from stdin.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print only the integer attribute count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parser = PersonAttributeParser()
    descriptions = [args.description] if args.description is not None else [
        line.strip() for line in sys.stdin if line.strip()
    ]
    if not descriptions:
        raise SystemExit("No description provided.")

    results = [parser.parse(description) for description in descriptions]
    if args.count_only:
        for result in results:
            print(result["count"])
        return
    for result in results:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
