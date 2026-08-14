#!/usr/bin/env python3
"""Archive root-level SFT dev artifacts into per-run dev subdirectories."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from person_sft_data import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "sft"
DEV_METRICS_PATTERN = re.compile(r"^dev_metrics_step_(\d+)\.json$")
DEV_PREDICTIONS_PATTERN = re.compile(r"^dev_predictions_step_(\d+)\.jsonl$")


def dev_artifact_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "dev"


def _is_root_dev_artifact(path: Path) -> bool:
    name = path.name
    return bool(
        DEV_METRICS_PATTERN.match(name) or DEV_PREDICTIONS_PATTERN.match(name)
    )


def _streaming_same_file(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    return sha256_file(left) == sha256_file(right)


def _root_dev_artifacts(run_dir: Path) -> List[Path]:
    return [
        path
        for path in sorted(Path(run_dir).iterdir(), key=lambda entry: entry.name)
        if path.is_file() and _is_root_dev_artifact(path)
    ]


def _archive_one_file(source: Path, destination: Path, dry_run: bool) -> Dict[str, Any]:
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Destination conflict at {destination}")
        if not _streaming_same_file(source, destination):
            raise ValueError(
                f"Destination conflict for {source.name}: {source} -> {destination}"
            )
        if not dry_run:
            source.unlink()
        return {
            "action": "verified",
            "source": str(source),
            "destination": str(destination),
        }
    if dry_run:
        return {
            "action": "move",
            "source": str(source),
            "destination": str(destination),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(source), str(destination))
    return {
        "action": "moved",
        "source": str(source),
        "destination": str(destination),
    }


def archive_run_dev_artifacts(run_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    actions: List[Dict[str, Any]] = []
    for source in _root_dev_artifacts(run_dir):
        destination = dev_artifact_dir(run_dir) / source.name
        actions.append(_archive_one_file(source, destination, dry_run=dry_run))
    summary = {
        "run_dir": str(run_dir.resolve()),
        "dry_run": dry_run,
        "root_files": len(actions),
        "moved": sum(action["action"] == "moved" for action in actions),
        "verified": sum(action["action"] == "verified" for action in actions),
        "planned": sum(action["action"] == "move" for action in actions),
        "conflicts": 0,
    }
    return summary


def archive_all_runs(artifacts_root: Path, dry_run: bool = False) -> Dict[str, Any]:
    artifacts_root = Path(artifacts_root)
    if not artifacts_root.is_dir():
        raise FileNotFoundError(f"Artifacts root does not exist: {artifacts_root}")

    run_summaries: List[Dict[str, Any]] = []
    totals = {
        "runs": 0,
        "root_files": 0,
        "moved": 0,
        "verified": 0,
        "planned": 0,
        "conflicts": 0,
    }
    for run_dir in sorted(artifacts_root.iterdir(), key=lambda entry: entry.name):
        if not run_dir.is_dir():
            continue
        summary = archive_run_dev_artifacts(run_dir, dry_run=dry_run)
        if summary["root_files"]:
            run_summaries.append(summary)
            for key in ("root_files", "moved", "verified", "planned", "conflicts"):
                totals[key] += summary[key]
    totals["runs"] = len(run_summaries)

    return {
        "artifacts_root": str(artifacts_root.resolve()),
        "dry_run": dry_run,
        "runs": run_summaries,
        "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = archive_all_runs(args.artifacts_root, dry_run=args.dry_run)
    for run in summary["runs"]:
        print(
            f"{Path(run['run_dir']).name}: "
            f"moved={run['moved']} verified={run['verified']} planned={run['planned']}",
            flush=True,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
