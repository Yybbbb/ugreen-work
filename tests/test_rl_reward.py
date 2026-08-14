import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "rl_reward.py"


def load_module():
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("rl_reward", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def attrs(**overrides):
    base = {
        "age_group": "adult", "gender": "male",
        "upper_garment": {"type": "jacket", "color": "blue", "length": "long-sleeve"},
        "lower_garment": {"type": "pants", "color": "black", "length": "long"},
        "shoes": {"type": "sneakers", "color": "white"},
        "head": {"accessories": "none", "hairstyle": "short", "hair_color": "black", "hair_length": "short"},
        "carried_items": {"handbag": "none", "backpack": "none"},
        "handheld_items": {"dangerous_item": "none", "mobile_phone": "no"},
        "extra": [],
    }
    base.update(overrides)
    return base


def sim_always_one(field, gt_value, gen_value):
    return 1.0


class LengthScoreTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_ideal_18_to_24_is_one(self):
        for n in (18, 20, 24):
            self.assertEqual(self.mod.length_score(n), 1.0, f"words={n}")

    def test_mild_outside_is_neg_half(self):
        for n in (12, 17, 25, 28):
            self.assertEqual(self.mod.length_score(n), -0.5, f"words={n}")

    def test_far_outside_is_neg_one(self):
        for n in (5, 11, 29, 60):
            self.assertEqual(self.mod.length_score(n), -1.0, f"words={n}")


class SentenceStructureTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_good_caption_scores_one(self):
        self.assertEqual(self.mod.sentence_structure_score("An adult male wears a blue jacket."), 1.0)

    def test_json_scores_zero(self):
        self.assertEqual(self.mod.sentence_structure_score('{"gender":"male"}'), 0.0)

    def test_multi_sentence_scores_zero(self):
        self.assertEqual(self.mod.sentence_structure_score("An adult male wears a blue jacket. He carries a bag."), 0.0)


class BackgroundPenaltyTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_detects_background_words(self):
        self.assertEqual(self.mod.background_penalty("An adult male stands in a street with a building behind."), 1.0)

    def test_clean_caption_zero(self):
        self.assertEqual(self.mod.background_penalty("An adult male wears a blue jacket."), 0.0)


class InvalidFormatTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_empty_is_invalid(self):
        self.assertTrue(self.mod.is_invalid_format(""))

    def test_json_is_invalid(self):
        self.assertTrue(self.mod.is_invalid_format('{"gender":"male"}'))

    def test_pipe_fields_invalid(self):
        self.assertTrue(self.mod.is_invalid_format("male|blue jacket|short hair"))

    def test_natural_caption_valid(self):
        self.assertFalse(self.mod.is_invalid_format("An adult male wears a blue jacket."))


class LexicalSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_exact_is_one(self):
        self.assertEqual(self.mod.lexical_similarity("upper_garment.color", "blue", "blue"), 1.0)

    def test_different_color_is_zero(self):
        self.assertEqual(self.mod.lexical_similarity("upper_garment.color", "blue", "red"), 0.0)

    def test_synonym_not_recognized_lexically(self):
        # lexical matching does NOT know white ~= light-colored
        self.assertEqual(self.mod.lexical_similarity("upper_garment.color", "white", "light-colored"), 0.0)


class ClassifySampleTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_all_match(self):
        c = self.mod.classify_sample(attrs(), attrs(), sim_always_one)
        self.assertEqual(c["n_fab"], 0)
        self.assertGreater(c["soft_tp"], 0)
        self.assertEqual(c["soft_fp"], 0.0)
        self.assertEqual(c["soft_fn"], 0.0)

    def test_fabrication_when_gt_unknown_gen_concrete(self):
        gt = attrs(head={"accessories": "none", "hairstyle": "short", "hair_color": "black", "hair_length": "short"})
        gen = attrs(head={"accessories": "glasses", "hairstyle": "short", "hair_color": "black", "hair_length": "short"})
        c = self.mod.classify_sample(gt, gen, sim_always_one)
        self.assertEqual(c["n_fab"], 1)

    def test_miss_when_gt_concrete_gen_unknown(self):
        gt = attrs(shoes={"type": "sneakers", "color": "white"})
        gen = attrs(shoes={"type": "sneakers", "color": "none"})
        c = self.mod.classify_sample(gt, gen, sim_always_one)
        self.assertGreater(c["soft_fn"], 0.0)

    def test_partial_similarity_gives_half(self):
        # one field compared with similarity 0.5 -> soft_tp += 0.5, soft_fp += 0.5, soft_fn += 0.5
        gt = attrs(upper_garment={"type": "jacket", "color": "blue", "length": "long-sleeve"})
        gen = attrs(upper_garment={"type": "jacket", "color": "blue", "length": "long-sleeve"})

        def sim(field, g, p):
            return 0.5 if field == "upper_garment.color" else 1.0

        c = self.mod.classify_sample(gt, gen, sim)
        # upper_garment.color contributes 0.5 to tp/fp/fn; the rest match fully
        self.assertAlmostEqual(c["soft_fp"], 0.5, places=6)
        self.assertAlmostEqual(c["soft_fn"], 0.5, places=6)

    def test_conflict_zero_similarity(self):
        gt = attrs(upper_garment={"type": "jacket", "color": "blue", "length": "long-sleeve"})
        gen = attrs(upper_garment={"type": "jacket", "color": "red", "length": "long-sleeve"})

        def sim(field, g, p):
            return 0.0 if field == "upper_garment.color" else 1.0

        c = self.mod.classify_sample(gt, gen, sim)
        # color conflict: tp += 0, fp += 1, fn += 1
        self.assertAlmostEqual(c["soft_fp"], 1.0, places=6)
        self.assertAlmostEqual(c["soft_fn"], 1.0, places=6)

    def test_extra_unmatched_is_fabrication(self):
        gt = attrs(extra=["silver watch"])
        gen = attrs(extra=["silver watch", "golden ring"])

        def sim(field, g, p):
            return 1.0 if g == p else 0.0

        c = self.mod.classify_sample(gt, gen, sim)
        self.assertEqual(c["n_fab"], 1)


class ComputeRewardTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        # 20 words -> length_score 1.0; single sentence, subject start -> structure 1.0; no bg
        self.caption = "An adult male with short dark hair and glasses wears a blue jacket over a white t-shirt and dark pants."

    def _counts(self, **kw):
        base = {"soft_tp": 10.0, "soft_fp": 0.0, "soft_fn": 0.0, "n_gen_assert": 10, "n_fab": 0}
        base.update(kw)
        return base

    def test_empty_caption_is_negative_one(self):
        self.assertEqual(self.mod.compute_reward("", self._counts()), -1.0)

    def test_json_format_is_negative_one(self):
        self.assertEqual(self.mod.compute_reward('{"gender":"male"}', self._counts()), -1.0)

    def test_all_match_positive(self):
        # F1=1.0 (+0.40), structure 1.0 (+0.10), length 1.0 (+0.10), bg 0
        r = self.mod.compute_reward(self.caption, self._counts())
        self.assertAlmostEqual(r, 0.40 + 0.10 + 0.10, places=6)

    def test_f1_from_soft_counts(self):
        # F1 = 2*6/(2*6+1+5) = 12/18 = 0.6667 -> 0.40*0.6667 + 0.10 + 0.10
        r = self.mod.compute_reward(self.caption, self._counts(soft_tp=6.0, soft_fp=1.0, soft_fn=5.0, n_gen_assert=12))
        self.assertAlmostEqual(r, 0.40 * (12.0 / 18.0) + 0.10 + 0.10, places=6)

    def test_fabrication_dominant_is_negative(self):
        r = self.mod.compute_reward(self.caption, self._counts(soft_tp=0.0, soft_fp=0.0, soft_fn=10.0, n_gen_assert=10, n_fab=10))
        self.assertLess(r, 0.0)

    def test_f1_zero_no_fab_is_format_only(self):
        r = self.mod.compute_reward(self.caption, self._counts(soft_tp=0.0, soft_fp=0.0, soft_fn=10.0, n_gen_assert=0, n_fab=0))
        self.assertAlmostEqual(r, 0.10 + 0.10, places=6)


if __name__ == "__main__":
    unittest.main()
