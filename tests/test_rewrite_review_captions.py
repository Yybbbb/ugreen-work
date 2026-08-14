import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "data" / "scripts" / "rewrite_review_captions.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("rewrite_review_captions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScriptPresenceTests(unittest.TestCase):
    def test_rewrite_script_exists(self):
        self.assertTrue(SCRIPT_PATH.is_file())


class AttributePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script_module()

    def test_filters_placeholders_and_extra_recursively(self):
        attributes = {
            "gender": "female",
            "age_group": "unknown",
            "upper_garment": {
                "type": "t-shirt",
                "color": "white",
                "length": "short-sleeve",
            },
            "lower_garment": {
                "type": "other",
                "color": "black",
                "length": "",
            },
            "head": {
                "accessories": "none",
                "hair_color": "brown",
            },
            "carried_items": {"backpack": "no", "handbag": None},
            "extra": ["glasses", "watch"],
        }

        self.assertEqual(
            self.mod.filter_meaningful_attributes(attributes),
            {
                "gender": "female",
                "upper_garment": {
                    "type": "t-shirt",
                    "color": "white",
                    "length": "short-sleeve",
                },
                "lower_garment": {"color": "black"},
                "head": {"hair_color": "brown"},
            },
        )

    def test_filters_placeholder_only_lists_and_preserves_meaningful_items(self):
        attributes = {
            "tags": ["none", "striped", "unknown"],
            "empty": [],
            "false_value": False,
            "true_value": True,
        }
        self.assertEqual(
            self.mod.filter_meaningful_attributes(attributes),
            {"tags": ["striped"], "true_value": True},
        )


class QwenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script_module()
        cls.tasks = [
            {
                "id": "scene/a.json#0",
                "original_caption": "A person in a red shirt.",
                "attributes": {
                    "gender": "female",
                    "upper_garment": {"color": "white", "type": "shirt"},
                },
            }
        ]

    def test_chat_payload_is_text_only_and_disables_thinking(self):
        payload = self.mod.build_chat_payload(
            self.tasks,
            model="Qwen36-35b-caption",
            temperature=0.2,
            top_p=0.8,
            max_tokens=4096,
        )
        serialized = json.dumps(payload)
        self.assertEqual(payload["model"], "Qwen36-35b-caption")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("image_url", serialized)
        self.assertIn("scene/a.json#0", serialized)
        self.assertIn("authoritative_attributes", serialized)
        self.assertIn("must not use negative", serialized.lower())

    def test_parse_rewrites_uses_ids_not_response_order(self):
        raw = json.dumps(
            {
                "rewrites": [
                    {"id": "b", "caption": "Caption B."},
                    {"id": "a", "caption": "Caption A."},
                ]
            }
        )
        self.assertEqual(
            self.mod.parse_rewrites(raw, {"a", "b"}),
            {"a": "Caption A.", "b": "Caption B."},
        )

    def test_parse_rewrites_rejects_missing_or_duplicate_ids(self):
        with self.assertRaises(self.mod.ResponseError):
            self.mod.parse_rewrites(
                '{"rewrites":[{"id":"a","caption":"One."}]}',
                {"a", "b"},
            )
        with self.assertRaises(self.mod.ResponseError):
            self.mod.parse_rewrites(
                '{"rewrites":[{"id":"a","caption":"One."},'
                '{"id":"a","caption":"Two."}]}',
                {"a"},
            )

    def test_validation_rejects_negative_or_malformed_output(self):
        original = "A person in a red shirt."
        self.assertIsNone(
            self.mod.validate_rewrite(original, "A woman in a white shirt.")
        )
        self.assertIsNotNone(
            self.mod.validate_rewrite(original, "A person without a backpack.")
        )
        self.assertIsNotNone(
            self.mod.validate_rewrite(original, "```A person in white.```")
        )
        self.assertIsNotNone(self.mod.validate_rewrite(original, ""))


class CacheAndMaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script_module()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input_root = self.root / "review"
        self.work_root = self.root / "work"
        self.relative = Path("scene") / "frame.json"
        self.source_path = self.input_root / self.relative
        self.source_path.parent.mkdir(parents=True)
        self.document = {
            "schema_version": "test-v1",
            "image": {"path": "/images/frame.jpg"},
            "crops": [
                {
                    "crop_index": 7,
                    "caption": "A person in a red shirt.",
                    "attributes": {
                        "gender": "female",
                        "upper_garment": {"color": "white", "type": "shirt"},
                        "extra": ["watch"],
                    },
                    "human_review": {"status": "reviewed"},
                },
                {
                    "crop_index": 9,
                    "caption": "A person in black trousers.",
                    "attributes": {"lower_garment": {"color": "black"}},
                },
            ],
        }
        self.source_path.write_text(
            json.dumps(self.document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.task = self.mod.build_caption_task(
            self.relative, 0, self.document["crops"][0]
        )

    def test_cache_matches_original_and_already_applied_caption(self):
        rewritten = "A woman in a white shirt."
        self.mod.save_cached_rewrite(
            self.work_root,
            self.task,
            rewritten,
            model="Qwen36-35b-caption",
        )

        original_hit = self.mod.load_cached_rewrite(
            self.work_root,
            self.relative,
            0,
            self.document["crops"][0],
            model="Qwen36-35b-caption",
        )
        self.assertEqual(original_hit.caption, rewritten)
        self.assertFalse(original_hit.applied)

        applied_crop = dict(self.document["crops"][0], caption=rewritten)
        applied_hit = self.mod.load_cached_rewrite(
            self.work_root,
            self.relative,
            0,
            applied_crop,
            model="Qwen36-35b-caption",
        )
        self.assertEqual(applied_hit.caption, rewritten)
        self.assertTrue(applied_hit.applied)

    def test_cache_is_invalidated_when_meaningful_attributes_change(self):
        self.mod.save_cached_rewrite(
            self.work_root,
            self.task,
            "A woman in a white shirt.",
            model="Qwen36-35b-caption",
        )
        changed_crop = json.loads(json.dumps(self.document["crops"][0]))
        changed_crop["attributes"]["upper_garment"]["color"] = "blue"
        self.assertIsNone(
            self.mod.load_cached_rewrite(
                self.work_root,
                self.relative,
                0,
                changed_crop,
                model="Qwen36-35b-caption",
            )
        )

    def test_materialization_changes_only_selected_caption(self):
        rewritten = "A woman in a white shirt."
        before = json.loads(self.source_path.read_text(encoding="utf-8"))
        failures = self.mod.materialize_rewrites(
            self.input_root, {self.task.task_id: self.task}, {self.task.task_id: rewritten}
        )
        after = json.loads(self.source_path.read_text(encoding="utf-8"))

        self.assertEqual(failures, {})
        self.assertEqual(after["crops"][0]["caption"], rewritten)
        self.assertEqual(
            after["crops"][1]["caption"], before["crops"][1]["caption"]
        )
        before["crops"][0].pop("caption")
        after["crops"][0].pop("caption")
        self.assertEqual(after, before)
        self.assertEqual(list(self.source_path.parent.glob("*.tmp")), [])

    def test_materialization_refuses_to_overwrite_concurrent_caption_change(self):
        changed = json.loads(self.source_path.read_text(encoding="utf-8"))
        changed["crops"][0]["caption"] = "A manually edited caption."
        self.source_path.write_text(json.dumps(changed), encoding="utf-8")

        failures = self.mod.materialize_rewrites(
            self.input_root,
            {self.task.task_id: self.task},
            {self.task.task_id: "A woman in a white shirt."},
        )

        current = json.loads(self.source_path.read_text(encoding="utf-8"))
        self.assertIn(self.task.task_id, failures)
        self.assertEqual(
            current["crops"][0]["caption"], "A manually edited caption."
        )


class ClientAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script_module()

    def make_task(self, name, caption="A person in a red shirt."):
        return self.mod.build_caption_task(
            Path("scene") / f"{name}.json",
            0,
            {
                "crop_index": 1,
                "caption": caption,
                "attributes": {
                    "gender": "female",
                    "upper_garment": {"color": "white", "type": "shirt"},
                },
            },
        )

    def test_batch_failure_falls_back_to_individual_requests(self):
        tasks = [self.make_task("a"), self.make_task("b")]
        calls = []

        def requester(payload, _config):
            user = json.loads(payload["messages"][1]["content"])
            calls.append(len(user["tasks"]))
            if len(user["tasks"]) > 1:
                raise OSError("batch unavailable")
            task_id = user["tasks"][0]["id"]
            return json.dumps(
                {
                    "rewrites": [
                        {"id": task_id, "caption": "A woman in a white shirt."}
                    ]
                }
            )

        config = self.mod.RequestConfig(retries=0, retry_sleep=0)
        result = self.mod.rewrite_batch(tasks, config, requester=requester)

        self.assertEqual(set(result.successes), {task.task_id for task in tasks})
        self.assertEqual(result.failures, {})
        self.assertEqual(calls, [2, 1, 1])

    def test_invalid_negative_output_is_not_a_success(self):
        task = self.make_task("a")

        def requester(payload, _config):
            task_id = json.loads(payload["messages"][1]["content"])["tasks"][0][
                "id"
            ]
            return json.dumps(
                {
                    "rewrites": [
                        {"id": task_id, "caption": "A woman without a backpack."}
                    ]
                }
            )

        result = self.mod.rewrite_batch(
            [task],
            self.mod.RequestConfig(retries=1, retry_sleep=0),
            requester=requester,
        )
        self.assertEqual(result.successes, {})
        self.assertIn(task.task_id, result.failures)

    def test_validation_retry_sends_feedback_to_preserve_original_details(self):
        task = self.make_task(
            "short",
            (
                "The person is wearing a purple short-sleeved top with a subtle "
                "pattern and a striped band near the hem, revealing a bare arm."
            ),
        )
        calls = []

        def requester(payload, _config):
            request_task = json.loads(payload["messages"][1]["content"])["tasks"][0]
            calls.append(request_task)
            if len(calls) == 1:
                caption = "The person is wearing a purple short-sleeved top."
            else:
                self.assertEqual(
                    request_task["previous_validation_error"],
                    "caption is implausibly short",
                )
                self.assertIn("Preserve original visual details", request_task["retry_instruction"])
                caption = (
                    "The person is wearing a purple short-sleeved top with a subtle "
                    "pattern and a striped band near the hem, revealing a bare arm."
                )
            return json.dumps(
                {"rewrites": [{"id": request_task["id"], "caption": caption}]}
            )

        result = self.mod.rewrite_batch(
            [task],
            self.mod.RequestConfig(retries=1, retry_sleep=0),
            requester=requester,
        )

        self.assertEqual(result.failures, {})
        self.assertEqual(
            result.successes[task.task_id],
            (
                "The person is wearing a purple short-sleeved top with a subtle "
                "pattern and a striped band near the hem, revealing a bare arm."
            ),
        )

    def test_pipeline_updates_success_and_preserves_failed_caption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "review"
            work_root = root / "work"
            source = input_root / "scene" / "frame.json"
            source.parent.mkdir(parents=True)
            payload = {
                "meta": {"keep": True},
                "crops": [
                    {
                        "crop_index": 1,
                        "caption": "A person in a red shirt.",
                        "attributes": {
                            "gender": "female",
                            "upper_garment": {"color": "white", "type": "shirt"},
                        },
                    },
                    {
                        "crop_index": 2,
                        "caption": "A person in dark trousers.",
                        "attributes": {"lower_garment": {"color": "black"}},
                    },
                    {
                        "crop_index": 3,
                        "caption": "A person near a doorway.",
                        "attributes": {"gender": "unknown", "extra": ["watch"]},
                    },
                ],
            }
            source.write_text(json.dumps(payload), encoding="utf-8")

            def requester(request_payload, _config):
                request_tasks = json.loads(
                    request_payload["messages"][1]["content"]
                )["tasks"]
                rewrites = []
                for request_task in request_tasks:
                    caption = (
                        "A woman in a white shirt."
                        if "position=0" in request_task["id"]
                        else "A person without a light garment."
                    )
                    rewrites.append({"id": request_task["id"], "caption": caption})
                return json.dumps({"rewrites": rewrites})

            config = self.mod.PipelineConfig(
                input_root=input_root,
                work_root=work_root,
                request=self.mod.RequestConfig(retries=0, retry_sleep=0),
                batch_size=8,
                workers=1,
            )
            stats = self.mod.run_pipeline(config, requester=requester)
            current = json.loads(source.read_text(encoding="utf-8"))

            self.assertEqual(stats.total_crops, 3)
            self.assertEqual(stats.skipped_no_attributes, 1)
            self.assertEqual(stats.updated, 1)
            self.assertEqual(stats.failed, 1)
            self.assertEqual(
                current["crops"][0]["caption"], "A woman in a white shirt."
            )
            self.assertEqual(
                current["crops"][1]["caption"],
                "A person in dark trousers.",
            )
            self.assertEqual(
                current["crops"][2]["caption"], "A person near a doorway."
            )
            self.assertEqual(current["meta"], {"keep": True})

    def test_audit_counts_json_crops_and_meaningful_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene").mkdir()
            (root / "scene" / "a.json").write_text(
                json.dumps(
                    {
                        "crops": [
                            {"caption": "One.", "attributes": {"gender": "female"}},
                            {
                                "caption": "Two.",
                                "attributes": {"gender": "unknown", "extra": ["hat"]},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stats = self.mod.audit_input(root)
            self.assertEqual(stats.json_count, 1)
            self.assertEqual(stats.crop_count, 2)
            self.assertEqual(stats.caption_count, 2)
            self.assertEqual(stats.meaningful_attribute_crops, 1)


if __name__ == "__main__":
    unittest.main()
