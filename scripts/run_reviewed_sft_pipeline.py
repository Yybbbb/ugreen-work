#!/usr/bin/env python3
"""Run reviewed-data V1 through V4B with overlapped Qwen evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from reviewed_sft_pipeline_config import (
    DEFAULT_PROJECT_ROOT,
    QWEN_BASE_URL,
    ExperimentSpec,
    StageSpec,
    experiment_specs,
    florence_stage_commands,
    gt_stage,
    qwen_stage_commands,
)


EXPECTED_REVIEWED = {
    "train": (30000, "4752fe4631a27d873bc053caff16ab712b4a09d23ee6ab1ce98fe14ee7d5c575"),
    "dev": (2000, "a725317ca6b3df1a58992975d02b6e9020b6ae9832c22652e23b2f9e3da4b7f9"),
    "test": (4328, "4beeb8dfd29917ffa787a0f84c30f3f3be77d82a1e172d23d61c11c7db82772c"),
    "rl": (5981, "d0b42e627e4e8913421915a6c27b3706aaf03bc582c5ff1c4fa1d97354859b60"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sanitize_and_validate_prepared(
    project_root: Path,
    expected: Mapping[str, tuple[int, str]] | None = None,
) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    prepared = root / "data" / "prepared"
    metadata_path = prepared / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test_path = prepared / "test.jsonl"
    manifest_path = root / "data" / "manifests" / "test.jsonl"
    test_rows = list(iter_jsonl(test_path))
    empty_ids = [str(row.get("sample_id") or "") for row in test_rows if not str(row.get("label") or "").strip()]
    if empty_ids:
        empty_set = set(empty_ids)
        clean_test = [row for row in test_rows if str(row.get("sample_id") or "") not in empty_set]
        clean_manifest = [row for row in iter_jsonl(manifest_path) if str(row.get("sample_id") or "") not in empty_set]
        write_jsonl_atomic(test_path, clean_test)
        write_jsonl_atomic(manifest_path, clean_manifest)
        split = metadata["splits"]["test"]
        split.update({
            "samples": len(clean_test),
            "sha256": sha256_file(test_path),
            "manifest_sha256": sha256_file(manifest_path),
        })
        metadata["test_empty_caption_cleanup"] = {"removed_ids": empty_ids, "updated_at": now()}
        write_json_atomic(metadata_path, metadata)

    result: Dict[str, Any] = {"removed_empty_caption_ids": empty_ids}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for split_name in ("train", "dev", "test", "rl"):
        split = metadata["splits"][split_name]
        path = Path(split["path"])
        rows = sum(1 for _ in iter_jsonl(path))
        digest = sha256_file(path)
        if rows != int(split["samples"]) or digest != split["sha256"]:
            raise ValueError(f"prepared {split_name} does not match metadata")
        result[f"{split_name}_samples"] = rows
        result[f"{split_name}_sha256"] = digest
        if expected is not None:
            expected_rows, expected_hash = expected[split_name]
            if rows != expected_rows or digest != expected_hash:
                raise ValueError(
                    f"prepared {split_name} is not the approved reviewed dataset: "
                    f"got {rows}/{digest}, expected {expected_rows}/{expected_hash}"
                )
    return result


def archive_previous_artifacts(
    project_root: Path,
    specs: Sequence[ExperimentSpec],
    state: MutableMapping[str, Any],
    state_path: Path,
    timestamp: str,
) -> Path:
    root = Path(project_root).resolve()
    archive = state.setdefault("archive", {})
    if "root" not in archive:
        archive["root"] = str(root / "artifacts" / "sft" / f"reviewed_retrain_backup_{timestamp}")
        archive["status"] = "running"
        archive["items"] = {}
        write_json_atomic(state_path, state)
    backup_root = Path(archive["root"])
    backup_root.mkdir(parents=True, exist_ok=True)
    if archive.get("status") == "complete":
        return backup_root

    sources = [(spec.run_dir, backup_root / spec.run_name) for spec in specs]
    prepared_backup = backup_root / "data_prepared"
    for name in ("test_qwen_attributes.jsonl", "test_qwen_attributes.manifest.json"):
        sources.append((root / "data" / "prepared" / name, prepared_backup / name))
    for source, destination in sources:
        key = str(source)
        item = archive["items"].setdefault(key, {"source": key, "destination": str(destination), "status": "pending"})
        if item.get("status") == "complete" or (destination.exists() and not source.exists()):
            item["status"] = "complete"
            continue
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"archive destination already exists: {destination}")
            item["status"] = "moving"
            write_json_atomic(state_path, state)
            shutil.move(str(source), str(destination))
        item["status"] = "complete"
        item["updated_at"] = now()
        write_json_atomic(state_path, state)
    archive["status"] = "complete"
    archive["completed_at"] = now()
    write_json_atomic(state_path, state)
    return backup_root


class StateStore:
    def __init__(self, path: Path, state: MutableMapping[str, Any]):
        self.path = path
        self.state = state
        self.lock = threading.Lock()

    def update_stage(self, key: str, payload: Mapping[str, Any]) -> None:
        with self.lock:
            self.state.setdefault("stages", {})[key] = dict(payload)
            write_json_atomic(self.path, self.state)


def wait_for_qwen(base_url: str = QWEN_BASE_URL, timeout: int = 1800) -> None:
    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/") + "/models"
    last_error = "service unavailable"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("data"):
                return
            last_error = "model list is empty"
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(10)
    raise RuntimeError(f"Qwen service unavailable after {timeout}s: {last_error}")


def resume_command(stage: StageSpec) -> Sequence[str]:
    command = list(stage.command)
    if stage.name != "train" or stage.completion_path is None:
        return command
    run_dir = stage.completion_path.parents[1]
    checkpoints = []
    for path in run_dir.glob("checkpoint-*") if run_dir.exists() else ():
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if (path / "training_state.pt").exists():
            checkpoints.append((step, path))
    if checkpoints:
        command.extend(["--resume-from", str(max(checkpoints)[1])])
    return command


def execute_stage(
    stage: StageSpec,
    store: StateStore,
    project_root: Path,
    runner=subprocess.run,
) -> None:
    key = f"{stage.experiment}:{stage.name}"
    previous = store.state.get("stages", {}).get(key, {})
    if previous.get("status") == "complete" and (
        stage.completion_path is None or stage.completion_path.exists()
    ):
        return
    command = list(resume_command(stage))
    stage.log_path.parent.mkdir(parents=True, exist_ok=True)
    store.update_stage(key, {"status": "running", "command": command, "started_at": now(), "log": str(stage.log_path)})
    environment = os.environ.copy()
    environment.update(stage.environment)
    try:
        with stage.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{now()}] START {' '.join(command)}\n")
            log.flush()
            runner(command, cwd=project_root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        if stage.completion_path is not None and not stage.completion_path.exists():
            raise RuntimeError(f"stage did not create completion path: {stage.completion_path}")
    except Exception as error:
        store.update_stage(key, {"status": "failed", "error": repr(error), "updated_at": now(), "log": str(stage.log_path)})
        raise
    store.update_stage(key, {"status": "complete", "updated_at": now(), "log": str(stage.log_path)})


def run_pipeline(project_root: Path, store: StateStore, test_samples: int) -> None:
    specs = experiment_specs(project_root)
    ready: queue.Queue[ExperimentSpec | None] = queue.Queue()
    qwen_failure: list[BaseException] = []

    def qwen_worker() -> None:
        try:
            wait_for_qwen()
            execute_stage(gt_stage(project_root), store, project_root)
            while True:
                spec = ready.get()
                if spec is None:
                    return
                for stage in qwen_stage_commands(spec, test_samples):
                    execute_stage(stage, store, project_root)
        except BaseException as error:
            qwen_failure.append(error)

    worker = threading.Thread(target=qwen_worker, name="qwen-evaluation-worker", daemon=False)
    worker.start()
    try:
        for spec in specs:
            if qwen_failure:
                raise RuntimeError("Qwen worker failed") from qwen_failure[0]
            for stage in florence_stage_commands(spec):
                execute_stage(stage, store, project_root)
            ready.put(spec)
        ready.put(None)
        worker.join()
        if qwen_failure:
            raise RuntimeError("Qwen worker failed") from qwen_failure[0]
    finally:
        if worker.is_alive():
            ready.put(None)
            worker.join()


def dry_run(project_root: Path) -> None:
    preflight = sanitize_and_validate_prepared(project_root, EXPECTED_REVIEWED)
    print(json.dumps({"preflight": preflight}, ensure_ascii=False))
    print(json.dumps({"stage": "qwen_gt:extract", "command": list(gt_stage(project_root).command)}, ensure_ascii=False))
    for spec in experiment_specs(project_root):
        for stage in (*florence_stage_commands(spec), *qwen_stage_commands(spec, preflight["test_samples"])):
            print(json.dumps({"stage": f"{stage.experiment}:{stage.name}", "command": list(stage.command), "completion": str(stage.completion_path)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.dry_run:
        dry_run(root)
        return

    pipeline_root = root / "artifacts" / "reviewed_five_run_pipeline"
    state_path = pipeline_root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"schema_version": "reviewed-five-run-v1", "created_at": now(), "stages": {}}
        write_json_atomic(state_path, state)
    store = StateStore(state_path, state)
    try:
        preflight = sanitize_and_validate_prepared(root, EXPECTED_REVIEWED)
        state["preflight"] = preflight
        write_json_atomic(state_path, state)
        archive_previous_artifacts(root, experiment_specs(root), state, state_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_pipeline(root, store, preflight["test_samples"])
        state["status"] = "complete"
        state["completed_at"] = now()
        write_json_atomic(state_path, state)
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        state["updated_at"] = now()
        write_json_atomic(state_path, state)
        raise


if __name__ == "__main__":
    main()
