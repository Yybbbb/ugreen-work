import importlib.util
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "merge_reviewed_person_dataset.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "merge_reviewed_person_dataset", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReviewIdentityTests(unittest.TestCase):
    def test_normalize_review_status_supports_current_legacy_and_aliases(self):
        mod = load_module()

        self.assertEqual(
            mod.normalize_review_status(
                {"human_review": {"status": "approved"}}
            ),
            "reviewed",
        )
        self.assertEqual(
            mod.normalize_review_status(
                {
                    "review": {
                        "status": "duplicate",
                        "updated_at": "2026-08-06T00:00:00Z",
                    }
                }
            ),
            "duplicate",
        )
        self.assertEqual(
            mod.normalize_review_status(
                {"review": {"status": "reviewed"}}
            ),
            "unreviewed",
        )
        self.assertEqual(
            mod.normalize_review_status(
                {"human_review": {"status": "needs_revision"}}
            ),
            "uncertain",
        )

    def test_crop_key_prefers_source_position_and_falls_back_to_index_bbox(self):
        mod = load_module()

        self.assertEqual(
            mod.crop_key({"source_crop_position": 3}, 7),
            ("position", 3),
        )
        self.assertEqual(
            mod.crop_key(
                {"crop_index": 2, "bbox_xyxy": [1, 2, 3, 4]}, 7
            ),
            ("index_bbox", 2, (1.0, 2.0, 3.0, 4.0)),
        )

    def test_frame_key_prefers_image_scene_path_for_regular_data(self):
        mod = load_module()
        record = {
            "image": {
                "scene_relative_path": "scene_a/frame_001.jpg",
            }
        }

        self.assertEqual(
            mod.frame_key(record, Path("different/frame.json")),
            "scene_a/frame_001",
        )


class RegularReviewActionTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def make_sample(self, name, caption="old"):
        return self.mod.SampleRecord(
            identity=(f"scene/{name}", ("index_position", 1, 0)),
            relative_json=Path(f"scene/{name}.json"),
            frame_payload={"image": {"scene_relative_path": f"scene/{name}.jpg"}},
            crop={
                "crop_index": 1,
                "caption": caption,
                "attributes": {"age": "adult"},
                "dimensions": {"scale_bin": "medium"},
            },
            source_id=None,
            is_xiaohongshu=False,
        )

    def test_overlay_uses_authoritative_fields_and_rewritten_caption(self):
        authoritative = {
            "caption": "human caption",
            "attributes": {"age": "adult", "hair": "black"},
            "dimensions": {"scale_bin": "large"},
            "human_review": {"status": "reviewed"},
        }
        rewritten = {
            "caption": "rewritten reviewed caption",
            "attributes": {"age": "wrong"},
            "dimensions": {"scale_bin": "tiny"},
        }

        merged = self.mod.overlay_rewritten_caption(authoritative, rewritten)

        self.assertEqual(merged["caption"], "rewritten reviewed caption")
        self.assertEqual(
            merged["attributes"], {"age": "adult", "hair": "black"}
        )
        self.assertEqual(merged["dimensions"], {"scale_bin": "large"})
        self.assertEqual(merged["review_status"], "reviewed")
        self.assertEqual(authoritative["caption"], "human caption")

    def test_regular_actions_move_reviewed_and_delete_test_bad_statuses(self):
        reviewed = self.make_sample("reviewed", "reviewed caption")
        rejected = self.make_sample("rejected")
        duplicate = self.make_sample("duplicate")
        kept = self.make_sample("kept")
        dataset = {
            split: {}
            for split in ("train", "test", "dev", "rl", "reserve")
        }
        for split in dataset:
            dataset[split][reviewed.identity] = copy.deepcopy(reviewed)
        dataset["test"][rejected.identity] = rejected
        dataset["test"][duplicate.identity] = duplicate
        dataset["test"][kept.identity] = kept
        reviews = [
            self.mod.ReviewedCrop("reviewed", reviewed),
            self.mod.ReviewedCrop("rejected", rejected),
            self.mod.ReviewedCrop("duplicate", duplicate),
        ]

        counts = self.mod.apply_regular_actions(dataset, reviews)

        for split in ("train", "dev", "rl", "reserve"):
            self.assertNotIn(reviewed.identity, dataset[split])
        self.assertIn(reviewed.identity, dataset["test"])
        self.assertEqual(
            dataset["test"][reviewed.identity].crop["review_status"],
            "reviewed",
        )
        self.assertNotIn(rejected.identity, dataset["test"])
        self.assertNotIn(duplicate.identity, dataset["test"])
        self.assertIn(kept.identity, dataset["test"])
        self.assertEqual(counts["regular_reviewed_to_test"], 1)
        self.assertEqual(counts["regular_rejected_removed_from_test"], 1)
        self.assertEqual(counts["regular_duplicate_removed_from_test"], 1)


class CaptionAllocationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def make_xhs(self, name, video, caption):
        return self.mod.SampleRecord(
            identity=(f"小红书/scene/{name}", ("position", 0)),
            relative_json=Path(f"小红书/scene/{video}/{name}.json"),
            frame_payload={
                "image": {
                    "dataset_id": "scene",
                    "source_id": video,
                    "artifact_key": f"videos/{video}/{name}",
                }
            },
            crop={"crop_index": 1, "caption": caption},
            source_id=video,
            is_xiaohongshu=True,
        )

    def test_caption_similarity_normalizes_case_punctuation(self):
        pairs = self.mod.high_similarity_pairs(
            [
                "An adult woman wears a black coat and carries a red bag.",
                "AN ADULT WOMAN wears a black coat, and carries a red bag!",
                "A young boy in a white shirt rides a bicycle outdoors.",
            ],
            threshold=0.90,
        )

        self.assertIn((0, 1), pairs)
        self.assertNotIn((0, 2), pairs)

    def test_allocator_keeps_video_and_similarity_components_together(self):
        samples = [
            self.make_xhs(
                "a1",
                "video_a",
                "An adult woman wears a black coat and carries a red bag.",
            ),
            self.make_xhs(
                "a2",
                "video_a",
                "A tall man wears a blue jacket with dark trousers.",
            ),
            self.make_xhs(
                "b1",
                "video_b",
                "An adult woman wears a black coat and carries a red bag.",
            ),
            self.make_xhs(
                "c1",
                "video_c",
                "A child in a yellow sweater holds a small toy.",
            ),
        ]

        allocation = self.mod.allocate_xiaohongshu(
            samples, regular_test_captions=[], threshold=0.90
        )

        self.assertEqual(allocation["video_a"], allocation["video_b"])
        self.assertIn(allocation["video_c"], {"train", "test"})
        self.assertEqual(
            allocation.train_crops + allocation.test_crops,
            len(samples),
        )

    def test_allocator_pins_regular_test_caption_conflict_to_test(self):
        caption = "An adult man wears a gray shirt and black trousers."
        samples = [self.make_xhs("a1", "video_a", caption)]

        allocation = self.mod.allocate_xiaohongshu(
            samples,
            regular_test_captions=[caption],
            threshold=0.90,
        )

        self.assertEqual(allocation["video_a"], "test")

    def test_global_similarity_space_prevents_threshold_flip(self):
        shorter = (
            "A child with short dark hair is wearing a light-colored "
            "long-sleeved top."
        )
        longer = (
            "A child with short dark hair is wearing a light-colored "
            "long-sleeved top and light-colored long pants."
        )
        fillers = [
            f"Adult person unique{i} wears black clothing and carries item."
            for i in range(50)
        ]
        similarity = self.mod.CaptionSimilarity([shorter, longer, *fillers])
        samples = [
            self.make_xhs("a1", "video_a", shorter),
            self.make_xhs("b1", "video_b", longer),
        ]

        allocation = self.mod.allocate_xiaohongshu(
            samples,
            regular_test_captions=[],
            threshold=0.90,
            similarity=similarity,
        )

        self.assertEqual(
            similarity.conflict_queries([shorter], [longer], threshold=0.90),
            {0},
        )
        self.assertEqual(allocation["video_a"], allocation["video_b"])


class LeakageAndSizingTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def make_sample(self, name, caption, reviewed=False, confidence=0.5, area=100):
        crop = {
            "crop_index": 1,
            "caption": caption,
            "confidence": confidence,
            "crop_area": area,
        }
        if reviewed:
            crop["review_status"] = "reviewed"
        return self.mod.SampleRecord(
            identity=(f"scene/{name}", ("index_position", 1, 0)),
            relative_json=Path(f"scene/{name}.json"),
            frame_payload={"image": {}},
            crop=crop,
            source_id=None,
            is_xiaohongshu=False,
        )

    def test_train_test_leakage_removes_only_non_reviewed_train_crop(self):
        caption = "An adult woman wears a red coat with black trousers."
        conflict = self.make_sample("conflict", caption)
        unrelated = self.make_sample(
            "unrelated", "A young man in a blue shirt carries a backpack."
        )
        test_sample = self.make_sample("test", caption, reviewed=True)
        train = {
            conflict.identity: conflict,
            unrelated.identity: unrelated,
        }

        removed = self.mod.remove_train_test_leakage(
            train,
            {test_sample.identity: test_sample},
            threshold=0.90,
        )

        self.assertEqual(removed, {conflict.identity})
        self.assertNotIn(conflict.identity, train)
        self.assertIn(unrelated.identity, train)

    def test_train_test_leakage_rejects_reviewed_to_reviewed_conflict(self):
        caption = "An adult woman wears a red coat with black trousers."
        train_sample = self.make_sample("train", caption, reviewed=True)
        test_sample = self.make_sample("test", caption, reviewed=True)

        with self.assertRaises(self.mod.ReviewedLeakageConflict):
            self.mod.remove_train_test_leakage(
                {train_sample.identity: train_sample},
                {test_sample.identity: test_sample},
                threshold=0.90,
            )

    def test_reduce_to_limit_never_removes_reviewed_crop(self):
        reviewed = self.make_sample("reviewed", "same caption", reviewed=True)
        low = self.make_sample(
            "low", "same caption", confidence=0.1, area=10
        )
        high = self.make_sample(
            "high", "different caption", confidence=0.9, area=500
        )
        samples = {
            sample.identity: sample for sample in (reviewed, low, high)
        }

        removed = self.mod.reduce_to_limit(samples, limit=2)

        self.assertIn(reviewed.identity, samples)
        self.assertEqual(removed, {low.identity})
        self.assertEqual(len(samples), 2)

    def test_validate_counts_allows_four_percent_and_rejects_more(self):
        self.mod.validate_counts(
            {"train": 28800},
            {"train": 30000},
            tolerance=0.04,
        )
        with self.assertRaises(self.mod.PublicationBlocked):
            self.mod.validate_counts(
                {"train": 28799},
                {"train": 30000},
                tolerance=0.04,
            )


class LoadingAndMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def write_frame(self, path, captions, image=None, statuses=None):
        image = image or {
            "scene_relative_path": path.with_suffix(".jpg").name,
            "path": str(path.with_suffix(".jpg")),
            "width": 100,
            "height": 100,
        }
        crops = []
        for position, caption in enumerate(captions):
            crop = {
                "crop_index": position + 1,
                "bbox_xyxy": [position, 0, position + 10, 20],
                "caption": caption,
                "confidence": 0.8,
                "crop_area": 200,
                "attributes": {"age": "adult"},
                "dimensions": {"scale_bin": "medium"},
            }
            if statuses:
                crop["human_review"] = {"status": statuses[position]}
            crops.append(crop)
        payload = {
            "image": image,
            "crops": crops,
            "person_crop_count": len(crops),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def test_load_and_materialize_preserves_path_and_filters_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            relative = Path("scene/frame.json")
            source_path = source / relative
            self.write_frame(source_path, ["first caption", "second caption"])
            source_bytes = source_path.read_bytes()

            samples = self.mod.load_split_samples(source, split="train")
            first_identity = next(
                identity
                for identity, sample in samples.items()
                if sample.crop["caption"] == "first caption"
            )
            samples.pop(first_identity)
            output = root / "staging" / "train"
            frames = self.mod.materialize_split(samples, output)

            written = json.loads(
                (output / relative).read_text(encoding="utf-8")
            )
            self.assertEqual(frames, 1)
            self.assertEqual(written["person_crop_count"], 1)
            self.assertEqual(written["crops"][0]["caption"], "second caption")
            self.assertEqual(source_path.read_bytes(), source_bytes)

    def test_load_reviewed_prefers_rewritten_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_root = root / "reviewed"
            rewrite_root = root / "rewrite"
            relative = Path("scene/frame.json")
            self.write_frame(
                review_root / relative,
                ["human caption"],
                statuses=["reviewed"],
            )
            self.write_frame(
                rewrite_root / relative,
                ["rewritten caption"],
                statuses=["reviewed"],
            )

            regular, xiaohongshu, counts = self.mod.load_reviewed_crops(
                review_root, rewrite_root
            )

            self.assertEqual(len(regular), 1)
            self.assertEqual(xiaohongshu, [])
            self.assertEqual(
                regular[0].sample.crop["caption"], "rewritten caption"
            )
            self.assertEqual(
                regular[0].sample.crop["review_status"], "reviewed"
            )
            self.assertEqual(counts["regular_reviewed"], 1)

    def test_load_xiaohongshu_reviewed_fills_expanded_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_root = root / "reviewed"
            rewrite_root = root / "rewrite"
            relative = Path(
                "小红书/dataset/annotations/profile/videos/video_a/frame.json"
            )
            image = {
                "dataset_id": "dataset",
                "source_id": "video_a",
                "artifact_key": "videos/video_a/frame",
                "path": "/tmp/frame.jpg",
                "width": 100,
                "height": 100,
            }
            for target, caption in (
                (review_root, "human caption"),
                (rewrite_root, "rewritten caption"),
            ):
                payload = self.write_frame(
                    target / relative,
                    [caption],
                    image=image,
                    statuses=["reviewed"],
                )
                payload["crops"][0]["source_crop_position"] = 0
                (target / relative).write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            _, xiaohongshu, _ = self.mod.load_reviewed_crops(
                review_root, rewrite_root
            )

            self.assertEqual(len(xiaohongshu), 1)
            self.assertEqual(
                xiaohongshu[0].crop["expanded_bbox_xyxy"],
                xiaohongshu[0].crop["bbox_xyxy"],
            )


class MergePipelineTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def sample(
        self,
        name,
        caption,
        *,
        reviewed=False,
        xhs=False,
        video=None,
        confidence=0.5,
    ):
        crop = {
            "crop_index": 1,
            "caption": caption,
            "confidence": confidence,
            "crop_area": 100,
            "attributes": {},
            "dimensions": {"scale_bin": "medium"},
        }
        if reviewed:
            crop["review_status"] = "reviewed"
        prefix = "小红书/scene" if xhs else "scene"
        return self.mod.SampleRecord(
            identity=(f"{prefix}/{name}", ("index_position", 1, 0)),
            relative_json=Path(f"{prefix}/{name}.json"),
            frame_payload={"image": {"source_id": video} if video else {}},
            crop=crop,
            source_id=video,
            is_xiaohongshu=xhs,
        )

    def test_merge_samples_applies_rules_and_preserves_rl_subset(self):
        reviewed = self.sample("regular_reviewed", "Reviewed regular caption")
        rejected = self.sample("regular_rejected", "Rejected caption")
        duplicate = self.sample("regular_duplicate", "Duplicate caption")
        train_rl = self.sample(
            "train_rl", "A person wears a green jacket.", confidence=0.01
        )
        train_a = self.sample("train_a", "A woman carries a violet handbag.")
        train_b = self.sample("train_b", "A man wears an orange sweater.")
        test_keep = self.sample("test_keep", "A child wears a yellow coat.")
        dev_keep = self.sample("dev_keep", "A person in a white shirt.")
        dataset = {
            "train": {
                sample.identity: sample
                for sample in (train_rl, train_a, train_b)
            },
            "test": {
                sample.identity: sample
                for sample in (reviewed, rejected, duplicate, test_keep)
            },
            "dev": {dev_keep.identity: dev_keep},
            "rl": {train_rl.identity: copy.deepcopy(train_rl)},
            "reserve": {},
        }
        regular = [
            self.mod.ReviewedCrop("reviewed", reviewed),
            self.mod.ReviewedCrop("rejected", rejected),
            self.mod.ReviewedCrop("duplicate", duplicate),
        ]
        xhs_train_or_test = [
            self.sample(
                "xhs_a",
                "An adult wears a silver patterned jacket.",
                reviewed=True,
                xhs=True,
                video="video_a",
            ),
            self.sample(
                "xhs_b",
                "A teenager carries a turquoise shoulder bag.",
                reviewed=True,
                xhs=True,
                video="video_b",
            ),
        ]

        report = self.mod.merge_samples(
            dataset,
            regular,
            xhs_train_or_test,
            review_status_counts={"regular_reviewed": 1},
            baselines={"train": 3, "test": 3, "dev": 1, "rl": 1},
            threshold=0.90,
            shortfall_tolerance=0.04,
        )

        self.assertIn(reviewed.identity, dataset["test"])
        self.assertNotIn(rejected.identity, dataset["test"])
        self.assertNotIn(duplicate.identity, dataset["test"])
        self.assertEqual(len(dataset["train"]), 3)
        self.assertEqual(len(dataset["test"]), 3)
        self.assertIn(train_rl.identity, dataset["train"])
        self.assertIn(train_rl.identity, dataset["rl"])
        xhs_locations = [
            split
            for split in ("train", "test")
            for sample in xhs_train_or_test
            if sample.identity in dataset[split]
        ]
        self.assertEqual(sorted(xhs_locations), ["test", "train"])
        self.assertEqual(report["after"]["train"], 3)
        self.assertEqual(report["after"]["test"], 3)

    def write_one_crop_split(self, root, split, caption, frame_name=None):
        frame_name = frame_name or split
        path = root / split / "scene" / f"{frame_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": {
                "scene_relative_path": f"scene/{frame_name}.jpg",
                "path": str(path.with_suffix(".jpg")),
                "width": 100,
                "height": 100,
            },
            "crops": [
                {
                    "crop_index": 1,
                    "bbox_xyxy": [0, 0, 10, 20],
                    "caption": caption,
                    "confidence": 0.8,
                    "crop_area": 200,
                    "attributes": {},
                    "dimensions": {"scale_bin": "medium"},
                }
            ],
            "person_crop_count": 1,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_dry_run_writes_report_without_modifying_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "data"
            captions = {
                "train": "A woman wears a red jacket.",
                "test": "A man carries a blue backpack.",
                "dev": "A child wears a yellow sweater.",
                "rl": "A woman wears a red jacket.",
                "reserve": "An adult wears a white shirt.",
            }
            paths = {}
            for split, caption in captions.items():
                frame_name = "train" if split == "rl" else split
                paths[split] = self.write_one_crop_split(
                    dataset_root, split, caption, frame_name=frame_name
                )
            original = {split: path.read_bytes() for split, path in paths.items()}
            review_root = root / "reviewed"
            rewrite_root = root / "rewrite"
            review_root.mkdir()
            rewrite_root.mkdir()
            config = self.mod.MergeConfig(
                dataset_root=dataset_root,
                review_root=review_root,
                rewrite_root=rewrite_root,
                threshold=0.90,
                shortfall_tolerance=0.04,
                baselines={"train": 1, "test": 1, "dev": 1, "rl": 1},
            )

            report_path = self.mod.run_merge(config, publish=False)

            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["after"]["train"], 1)
            self.assertEqual(report["validation_errors"], 0)
            for split, path in paths.items():
                self.assertEqual(path.read_bytes(), original[split])

    def test_publish_materializes_splits_manifests_and_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "data"
            captions = {
                "train": "A woman wears a red jacket.",
                "test": "A man carries a blue backpack.",
                "dev": "A child wears a yellow sweater.",
                "rl": "A woman wears a red jacket.",
                "reserve": "An adult wears a white shirt.",
            }
            for split, caption in captions.items():
                self.write_one_crop_split(
                    dataset_root,
                    split,
                    caption,
                    frame_name="train" if split == "rl" else split,
                )
            review_root = root / "reviewed"
            rewrite_root = root / "rewrite"
            review_root.mkdir()
            rewrite_root.mkdir()
            config = self.mod.MergeConfig(
                dataset_root=dataset_root,
                review_root=review_root,
                rewrite_root=rewrite_root,
                threshold=0.90,
                shortfall_tolerance=0.04,
                baselines={"train": 1, "test": 1, "dev": 1, "rl": 1},
            )

            report_path = self.mod.run_merge(config, publish=True)

            self.assertTrue(report_path.is_file())
            self.assertTrue((dataset_root / "manifests" / "train.jsonl").is_file())
            self.assertTrue((dataset_root / "split_statistics.json").is_file())
            manifest_rows = (
                dataset_root / "manifests" / "train.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(
                json.loads(manifest_rows[0])["sample_id"],
                "scene/train.json#0",
            )
            self.assertFalse(
                (dataset_root / ".reviewed_merge_work" / "backup").exists()
            )
            self.assertTrue(
                (
                    dataset_root
                    / ".reviewed_merge_work"
                    / "caption_similarity.joblib"
                ).is_file()
            )
            validation = self.mod.validate_published(config)
            self.assertEqual(validation["counts"]["train"], 1)
            self.assertEqual(validation["validation_errors"], 0)

    def test_parse_args_requires_one_execution_mode(self):
        args = self.mod.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--review-root",
                "/tmp/reviewed",
                "--rewrite-root",
                "/tmp/rewrite",
                "--dry-run",
            ]
        )

        self.assertTrue(args.dry_run)
        self.assertFalse(args.publish)


if __name__ == "__main__":
    unittest.main()
