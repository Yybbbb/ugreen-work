from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "eval/scripts/benchmark_crop_count_sequential.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_crop_count_sequential", SCRIPT_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class BenchmarkCropCountSequentialTests(unittest.TestCase):
    def test_parse_bbox_tokens_preserves_prompt_order(self) -> None:
        prompt = (
            "<REGIONS_TO_DESCRIPTIONS>"
            "<loc_1><loc_2><loc_3><loc_4><sep>"
            "<loc_5><loc_6><loc_7><loc_8>"
        )

        bbox_tokens = benchmark.parse_bbox_tokens(prompt)

        self.assertEqual(
            bbox_tokens,
            (
                "<loc_1><loc_2><loc_3><loc_4>",
                "<loc_5><loc_6><loc_7><loc_8>",
            ),
        )
        self.assertEqual(
            benchmark.build_single_prompts(bbox_tokens),
            (
                "<REGION_TO_DESCRIPTION><loc_1><loc_2><loc_3><loc_4>",
                "<REGION_TO_DESCRIPTION><loc_5><loc_6><loc_7><loc_8>",
            ),
        )

    def test_full_dataset_crop_counts_match_expected_values(self) -> None:
        samples = benchmark.load_samples(
            benchmark.DEFAULT_DATA_DIR,
            ("train", "test"),
            limit_per_crop_count=0,
        )

        self.assertEqual(benchmark.count_samples(samples), benchmark.EXPECTED_FULL_COUNTS)
        self.assertEqual(len(samples), 8324)
        self.assertEqual(sum(sample.crop_count for sample in samples), 23427)

    def test_encoder_decoder_output_count_excludes_decoder_start_token(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True))
        generated_ids = torch.zeros((1, 321), dtype=torch.long)

        output_tokens = benchmark.generated_token_count(model, generated_ids, input_tokens=30)

        self.assertEqual(output_tokens, 320)

    def test_single_results_are_strictly_summed(self) -> None:
        calls = [
            benchmark.GenerationResult(
                generate_time_s=0.1,
                end_to_end_time_s=0.2,
                input_tokens=10,
                output_tokens=20,
                ended_with_eos=True,
                hit_max_new_tokens=False,
                peak_allocated_mib=600.0,
                peak_reserved_mib=700.0,
                raw_output="first",
            ),
            benchmark.GenerationResult(
                generate_time_s=0.3,
                end_to_end_time_s=0.4,
                input_tokens=11,
                output_tokens=320,
                ended_with_eos=False,
                hit_max_new_tokens=True,
                peak_allocated_mib=610.0,
                peak_reserved_mib=710.0,
                raw_output="second",
            ),
        ]

        result = benchmark.aggregate_single_results(calls)

        self.assertEqual(result["call_count"], 2)
        self.assertAlmostEqual(result["generate_time_s"], 0.4)
        self.assertAlmostEqual(result["end_to_end_time_s"], 0.6)
        self.assertEqual(result["input_tokens"], 21)
        self.assertEqual(result["output_tokens"], 340)
        self.assertEqual(result["ended_with_eos_count"], 1)
        self.assertEqual(result["hit_max_new_tokens_count"], 1)
        self.assertEqual(result["peak_allocated_mib"], 610.0)
        self.assertEqual(result["peak_reserved_mib"], 710.0)

    def test_summary_reports_single_over_multi_speedup(self) -> None:
        records = [
            {
                "crop_count": 2,
                "multi": {
                    "generate_time_s": 1.0,
                    "end_to_end_time_s": 1.2,
                    "output_tokens": 100,
                    "hit_max_new_tokens": False,
                    "peak_allocated_mib": 600.0,
                    "peak_reserved_mib": 700.0,
                },
                "single": {
                    "generate_time_s": 2.0,
                    "end_to_end_time_s": 2.4,
                    "output_tokens": 100,
                    "hit_max_new_tokens_count": 0,
                    "peak_allocated_mib": 610.0,
                    "peak_reserved_mib": 710.0,
                },
            }
        ]

        summary = benchmark.summarize_records(records)

        self.assertEqual(
            summary["by_crop_count"]["2"]["generate"][
                "aggregate_speedup_single_over_multi"
            ],
            2.0,
        )
        self.assertEqual(summary["by_crop_count"]["2"]["image_count"], 1)
        self.assertEqual(summary["by_crop_count"]["2"]["crop_count"], 2)

    def test_report_explains_chinese_metric_definitions(self) -> None:
        records = [
            {
                "crop_count": 2,
                "multi": {
                    "generate_time_s": 1.0,
                    "end_to_end_time_s": 1.2,
                    "output_tokens": 100,
                    "hit_max_new_tokens": False,
                    "peak_allocated_mib": 600.0,
                    "peak_reserved_mib": 700.0,
                },
                "single": {
                    "generate_time_s": 2.0,
                    "end_to_end_time_s": 2.4,
                    "output_tokens": 100,
                    "hit_max_new_tokens_count": 0,
                    "peak_allocated_mib": 610.0,
                    "peak_reserved_mib": 710.0,
                },
            }
        ]
        summary = benchmark.summarize_records(records)
        metadata = {
            "checkpoint": "/tmp/checkpoint",
            "data_dir": "/tmp/data",
            "splits": ["train", "test"],
            "device": "cuda:0",
            "cuda_visible_devices": "2",
            "dtype": "bf16",
            "num_beams": 1,
            "max_new_tokens": 320,
            "gpu_state": {},
        }

        report = benchmark.build_report(summary, metadata)

        self.assertIn("# 多 Crop 联合推理与单框顺序推理速度测试报告", report)
        self.assertIn("Generate-only（纯生成耗时）", report)
        self.assertIn("End-to-end（单次推理端到端耗时）", report)
        self.assertIn("Images/s", report)
        self.assertIn("Crops/s", report)
        self.assertIn("Tokens/s", report)
        self.assertIn("单框顺序模式总耗时 ÷ 联合模式总耗时", report)


if __name__ == "__main__":
    unittest.main()
