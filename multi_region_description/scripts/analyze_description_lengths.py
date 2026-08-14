#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOCATION_GROUP_RE = re.compile(r"(?:<loc_\d+>){4}")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare per-region description length distributions in two JSONL datasets."
    )
    parser.add_argument("cleaned_dir", type=Path)
    parser.add_argument("original_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def split_descriptions(label):
    descriptions = []
    cursor = 0
    for match in LOCATION_GROUP_RE.finditer(label):
        descriptions.append(label[cursor : match.start()].strip())
        cursor = match.end()
    if label[cursor:].strip():
        raise ValueError(f"Unexpected label suffix: {label[cursor:cursor + 80]!r}")
    return descriptions


def read_dataset(dataset_dir, dataset_name):
    rows = []
    records = {}
    for split in ("train", "test"):
        path = dataset_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                record = json.loads(line)
                descriptions = split_descriptions(record["label"])
                expected_regions = record["prompt"].count("<sep>") + 1
                if len(descriptions) != expected_regions:
                    raise ValueError(
                        f"{path}:{line_number}: parsed {len(descriptions)} descriptions, "
                        f"but prompt contains {expected_regions} regions"
                    )
                key = (split, line_number)
                records[key] = {
                    "image": record["image"],
                    "prompt": record["prompt"],
                    "descriptions": descriptions,
                }
                for region_index, description in enumerate(descriptions):
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "line_number": line_number,
                            "region_index": region_index,
                            "description": description,
                            "words": len(WORD_RE.findall(description)),
                            "characters": len(description),
                        }
                    )
    return rows, records


def percentile(values, quantile):
    return float(np.quantile(values, quantile, method="linear"))


def describe(values):
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "p05": percentile(array, 0.05),
        "p25": percentile(array, 0.25),
        "median": percentile(array, 0.50),
        "p75": percentile(array, 0.75),
        "p95": percentile(array, 0.95),
        "max": float(array.max()),
    }


def empirical_cdf_distance(first, second):
    first = np.sort(np.asarray(first, dtype=float))
    second = np.sort(np.asarray(second, dtype=float))
    support = np.unique(np.concatenate([first, second]))
    first_cdf = np.searchsorted(first, support, side="right") / len(first)
    second_cdf = np.searchsorted(second, support, side="right") / len(second)
    ks = float(np.max(np.abs(first_cdf - second_cdf)))
    if len(support) == 1:
        wasserstein = 0.0
    else:
        wasserstein = float(
            np.sum(np.abs(first_cdf[:-1] - second_cdf[:-1]) * np.diff(support))
        )
    return ks, wasserstein


def discrete_distribution_distances(first, second):
    first_counts = Counter(first)
    second_counts = Counter(second)
    support = sorted(set(first_counts) | set(second_counts))
    first_prob = np.array([first_counts[value] / len(first) for value in support])
    second_prob = np.array([second_counts[value] / len(second) for value in support])
    midpoint = (first_prob + second_prob) / 2

    def kl_divergence(probability, reference):
        mask = probability > 0
        return float(np.sum(probability[mask] * np.log2(probability[mask] / reference[mask])))

    js = 0.5 * kl_divergence(first_prob, midpoint) + 0.5 * kl_divergence(
        second_prob, midpoint
    )
    total_variation = 0.5 * float(np.sum(np.abs(first_prob - second_prob)))
    return js, total_variation


def compare_distributions(cleaned, original):
    cleaned_array = np.asarray(cleaned, dtype=float)
    original_array = np.asarray(original, dtype=float)
    ks, wasserstein = empirical_cdf_distance(cleaned_array, original_array)
    js, total_variation = discrete_distribution_distances(cleaned, original)
    pooled_std = math.sqrt(
        ((len(cleaned_array) - 1) * cleaned_array.var(ddof=1)
         + (len(original_array) - 1) * original_array.var(ddof=1))
        / (len(cleaned_array) + len(original_array) - 2)
    )
    mean_difference = float(cleaned_array.mean() - original_array.mean())
    return {
        "cleaned_minus_original_mean": mean_difference,
        "mean_ratio": float(cleaned_array.mean() / original_array.mean()),
        "cohens_d": mean_difference / pooled_std if pooled_std else 0.0,
        "ks_statistic": ks,
        "wasserstein_distance": wasserstein,
        "jensen_shannon_divergence_bits": js,
        "total_variation_distance": total_variation,
    }


def validate_and_pair(cleaned_records, original_records, cleaned_rows, original_rows):
    if cleaned_records.keys() != original_records.keys():
        raise ValueError("The datasets do not contain the same train/test record keys")
    for key in cleaned_records:
        cleaned = cleaned_records[key]
        original = original_records[key]
        if cleaned["image"] != original["image"] or cleaned["prompt"] != original["prompt"]:
            raise ValueError(f"Record mismatch at {key}")
        if len(cleaned["descriptions"]) != len(original["descriptions"]):
            raise ValueError(f"Region count mismatch at {key}")

    cleaned_lookup = {
        (row["split"], row["line_number"], row["region_index"]): row
        for row in cleaned_rows
    }
    original_lookup = {
        (row["split"], row["line_number"], row["region_index"]): row
        for row in original_rows
    }
    if cleaned_lookup.keys() != original_lookup.keys():
        raise ValueError("The datasets do not contain the same region keys")
    paired = []
    for key in cleaned_lookup:
        cleaned = cleaned_lookup[key]
        original = original_lookup[key]
        paired.append(
            {
                "split": key[0],
                "line_number": key[1],
                "region_index": key[2],
                "cleaned_words": cleaned["words"],
                "original_words": original["words"],
                "word_difference": cleaned["words"] - original["words"],
                "cleaned_characters": cleaned["characters"],
                "original_characters": original["characters"],
                "character_difference": cleaned["characters"] - original["characters"],
            }
        )
    return paired


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(cleaned_rows, original_rows, metric):
    counters = {
        "cleaned": Counter(row[metric] for row in cleaned_rows),
        "original": Counter(row[metric] for row in original_rows),
    }
    totals = {name: sum(counter.values()) for name, counter in counters.items()}
    support = sorted(set(counters["cleaned"]) | set(counters["original"]))
    return [
        {
            metric: value,
            "cleaned_count": counters["cleaned"][value],
            "cleaned_ratio": counters["cleaned"][value] / totals["cleaned"],
            "original_count": counters["original"][value],
            "original_ratio": counters["original"][value] / totals["original"],
            "ratio_difference": (
                counters["cleaned"][value] / totals["cleaned"]
                - counters["original"][value] / totals["original"]
            ),
        }
        for value in support
    ]


def plot_distributions(cleaned_rows, original_rows, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"cleaned": "#2a9d8f", "original": "#e76f51"}
    for column, metric in enumerate(("words", "characters")):
        cleaned = np.array([row[metric] for row in cleaned_rows])
        original = np.array([row[metric] for row in original_rows])
        minimum = min(cleaned.min(), original.min())
        maximum = max(cleaned.max(), original.max())
        if metric == "words":
            bins = np.arange(minimum - 0.5, maximum + 1.5, 1)
        else:
            bins = np.linspace(minimum, maximum, 45)
        axes[0, column].hist(
            original, bins=bins, density=True, alpha=0.55,
            label="original", color=colors["original"]
        )
        axes[0, column].hist(
            cleaned, bins=bins, density=True, alpha=0.55,
            label="cleaned", color=colors["cleaned"]
        )
        axes[0, column].set_title(f"{metric.capitalize()} distribution")
        axes[0, column].set_xlabel(metric)
        axes[0, column].set_ylabel("density")
        axes[0, column].legend()

        for name, values in (("original", original), ("cleaned", cleaned)):
            sorted_values = np.sort(values)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            axes[1, column].step(
                sorted_values, cdf, where="post", label=name, color=colors[name]
            )
        axes[1, column].set_title(f"{metric.capitalize()} empirical CDF")
        axes[1, column].set_xlabel(metric)
        axes[1, column].set_ylabel("cumulative probability")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def format_number(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def write_report(path, summaries, comparisons, paired, split_counts):
    lines = [
        "# Description 长度分布对比",
        "",
        "## 统计口径",
        "",
        "- 单位：每个 region 对应的一条 description。",
        "- words：按英文单词计数，连字符词（如 `t-shirt`）计为一个词。",
        "- characters：去除首尾空白后，包含空格和标点的字符数。",
        "- `<loc_*>` 坐标 token 不计入 description 长度。",
        f"- 样本数：train {split_counts['train']} 条 region，test {split_counts['test']} 条 region，共 {len(paired)} 条 region。",
        "",
        "## 描述性统计",
        "",
        "| metric | dataset | count | mean | std | min | p05 | p25 | median | p75 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in ("words", "characters"):
        for dataset in ("cleaned", "original"):
            stats = summaries[metric][dataset]
            lines.append(
                "| " + " | ".join(
                    [metric, dataset]
                    + [format_number(stats[key]) for key in (
                        "count", "mean", "std", "min", "p05", "p25",
                        "median", "p75", "p95", "max"
                    )]
                ) + " |"
            )
    lines.extend(
        [
            "",
            "## 分布差距",
            "",
            "| metric | 均值差 cleaned-original | 均值比 | Cohen's d | KS | Wasserstein | JS divergence (bits) | Total variation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in ("words", "characters"):
        comparison = comparisons[metric]
        lines.append(
            f"| {metric} | {comparison['cleaned_minus_original_mean']:.4f} | "
            f"{comparison['mean_ratio']:.4f} | {comparison['cohens_d']:.4f} | "
            f"{comparison['ks_statistic']:.4f} | {comparison['wasserstein_distance']:.4f} | "
            f"{comparison['jensen_shannon_divergence_bits']:.4f} | "
            f"{comparison['total_variation_distance']:.4f} |"
        )

    word_differences = np.array([row["word_difference"] for row in paired])
    character_differences = np.array([row["character_difference"] for row in paired])
    lines.extend(
        [
            "",
            "## 同 region 配对差值",
            "",
            "| metric | cleaned 更长 | 等长 | cleaned 更短 | 平均差值 | 差值中位数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric, differences in (("words", word_differences), ("characters", character_differences)):
        lines.append(
            f"| {metric} | {(differences > 0).mean():.2%} | {(differences == 0).mean():.2%} | "
            f"{(differences < 0).mean():.2%} | {differences.mean():.4f} | "
            f"{np.median(differences):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 文件说明",
            "",
            "- `description_length_distribution.png`：直方图与经验累计分布。",
            "- `word_length_distribution.csv`：每个词数长度的频数、比例和比例差。",
            "- `character_length_distribution.csv`：每个字符长度的频数、比例和比例差。",
            "- `paired_length_differences.csv`：同一 region 的 cleaned-original 配对差值。",
            "- `summary.json`：全部描述性统计和分布距离指标。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_rows, cleaned_records = read_dataset(args.cleaned_dir, "cleaned")
    original_rows, original_records = read_dataset(args.original_dir, "original")
    paired = validate_and_pair(
        cleaned_records, original_records, cleaned_rows, original_rows
    )

    summaries = {}
    comparisons = {}
    for metric in ("words", "characters"):
        cleaned_values = [row[metric] for row in cleaned_rows]
        original_values = [row[metric] for row in original_rows]
        summaries[metric] = {
            "cleaned": describe(cleaned_values),
            "original": describe(original_values),
        }
        comparisons[metric] = compare_distributions(cleaned_values, original_values)

        rows = distribution_rows(cleaned_rows, original_rows, metric)
        write_csv(
            args.output_dir / f"{metric[:-1] if metric.endswith('s') else metric}_length_distribution.csv",
            rows,
            list(rows[0]),
        )

    write_csv(
        args.output_dir / "paired_length_differences.csv",
        paired,
        list(paired[0]),
    )
    plot_distributions(
        cleaned_rows, original_rows, args.output_dir / "description_length_distribution.png"
    )
    split_counts = Counter(row["split"] for row in cleaned_rows)
    payload = {
        "region_counts": dict(split_counts),
        "summaries": summaries,
        "distribution_comparisons": comparisons,
        "paired_differences": {
            metric: describe([row[f"{metric}_difference"] for row in paired])
            for metric in ("word", "character")
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        args.output_dir / "REPORT.md",
        summaries,
        comparisons,
        paired,
        split_counts,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
