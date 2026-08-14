#!/usr/bin/env python3
"""Run both V4 full-training experiments and their evaluations serially."""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

from v4_serial_config import DEFAULT_PROJECT_ROOT, StageSpec, serial_stage_specs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _wait_for_qwen(base_url: str = "http://127.0.0.1:6097/v1", timeout: int = 1800) -> None:
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
    raise RuntimeError(f"Qwen service did not become ready within {timeout}s: {last_error}")


def _resume_command(stage: StageSpec) -> Sequence[str]:
    command = list(stage.command)
    if stage.name != "train" or stage.completion_path is None:
        return command
    run_dir = stage.completion_path.parents[1]
    candidates = []
    for path in run_dir.glob("checkpoint-*") if run_dir.exists() else ():
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if (path / "training_state.pt").exists():
            candidates.append((step, path))
    if candidates and "--resume-from" not in command:
        command.extend(["--resume-from", str(max(candidates)[1])])
    return command


def execute_stages(
    stages: Iterable[StageSpec],
    state: MutableMapping[str, Any],
    *,
    runner: Callable[..., Any] = subprocess.run,
    state_path: Path | None = None,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> None:
    stage_state = state.setdefault("stages", {})
    for stage in stages:
        key = f"{stage.experiment}:{stage.name}"
        if stage.completion_path is not None and stage.completion_path.exists():
            stage_state[key] = {"status": "complete", "skipped": True, "updated_at": _now()}
            if state_path is not None:
                _write_state(state_path, state)
            print(f"SKIP {key}: {stage.completion_path}", flush=True)
            continue
        if stage.name == "extract":
            _wait_for_qwen()
        command = _resume_command(stage)
        stage.log_path.parent.mkdir(parents=True, exist_ok=True)
        stage_state[key] = {"status": "running", "command": command, "started_at": _now()}
        if state_path is not None:
            _write_state(state_path, state)
        print(f"START {key}: {' '.join(command)}", flush=True)
        environment = os.environ.copy()
        environment.update(stage.environment)
        try:
            with stage.log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{_now()}] START {' '.join(command)}\n")
                log.flush()
                runner(
                    command,
                    cwd=project_root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except Exception as error:
            stage_state[key] = {"status": "failed", "error": repr(error), "updated_at": _now()}
            if state_path is not None:
                _write_state(state_path, state)
            print(f"FAILED {key}: {error}", flush=True)
            raise
        if stage.completion_path is not None and not stage.completion_path.exists():
            error = RuntimeError(
                f"stage {key} exited successfully but did not create {stage.completion_path}"
            )
            stage_state[key] = {"status": "failed", "error": str(error), "updated_at": _now()}
            if state_path is not None:
                _write_state(state_path, state)
            raise error
        stage_state[key] = {"status": "complete", "log": str(stage.log_path), "updated_at": _now()}
        if state_path is not None:
            _write_state(state_path, state)
        print(f"DONE {key}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    stages = serial_stage_specs(root)
    if args.dry_run:
        for stage in stages:
            print(json.dumps({
                "experiment": stage.experiment,
                "stage": stage.name,
                "command": list(stage.command),
                "environment": stage.environment,
                "completion": str(stage.completion_path) if stage.completion_path else None,
            }, ensure_ascii=False))
        return
    pipeline_root = root / "artifacts" / "v4_serial_pipeline"
    state_path = pipeline_root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"schema_version": "v4-serial-pipeline-v1", "created_at": _now(), "stages": {}}
    execute_stages(stages, state, state_path=state_path, project_root=root)
    state["status"] = "complete"
    state["completed_at"] = _now()
    _write_state(state_path, state)
    print("V4 serial pipeline complete", flush=True)


if __name__ == "__main__":
    main()
