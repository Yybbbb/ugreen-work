#!/usr/bin/env python3
"""Inspect or start the fixed vLLM services used by Florence RL rewards.

This command deliberately has no replace, stop, or remove operation. An existing
container is either reused after a full HTTP preflight or left untouched.
"""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from rl_clients import preflight_service
from rl_service_config import (
    EXTRACTOR_SERVICE,
    JUDGE_SERVICE,
    SERVICES,
    ServiceSpec,
    docker_run_argv,
)


def selected_services(target: str) -> tuple[ServiceSpec, ...]:
    if target == "all":
        return SERVICES
    selected = tuple(spec for spec in SERVICES if spec.role == target)
    if not selected:
        raise ValueError(f"unknown service target: {target!r}")
    return selected


def print_command(spec: ServiceSpec) -> str:
    return shlex.join(docker_run_argv(spec))


def container_state(
    spec: ServiceSpec,
    *,
    run_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Optional[str]:
    result = run_fn(
        ["docker", "inspect", "--format", "{{.State.Status}}", spec.container_name],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def check_service(
    spec: ServiceSpec,
    *,
    container_state_fn: Callable[[ServiceSpec], Optional[str]] = container_state,
    preflight_fn: Callable[..., Dict[str, str]] = preflight_service,
) -> Dict[str, Any]:
    state = container_state_fn(spec)
    if state != "running":
        raise RuntimeError(
            f"{spec.role} container state is {state or 'missing'}: {spec.container_name}"
        )
    probe = preflight_fn(spec.base_url, spec.model)
    return {"role": spec.role, "state": state, "probe": probe}


def start_service(
    spec: ServiceSpec,
    *,
    container_state_fn: Callable[[ServiceSpec], Optional[str]] = container_state,
    run_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    preflight_fn: Callable[..., Dict[str, str]] = preflight_service,
    timeout: float = 1800.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    initial_state = container_state_fn(spec)
    if initial_state is not None:
        try:
            result = check_service(
                spec,
                container_state_fn=lambda unused: initial_state,
                preflight_fn=preflight_fn,
            )
        except Exception as exc:
            raise RuntimeError(
                f"existing {spec.container_name} is unhealthy and will not be replaced: {exc}"
            ) from exc
        return {**result, "action": "reused"}

    if not Path(spec.model_path).is_dir():
        raise RuntimeError(f"model directory does not exist: {spec.model_path}")
    run_fn(docker_run_argv(spec), check=True, text=True, capture_output=True)

    deadline = monotonic_fn() + timeout
    last_error = "service has not reached running state"
    while True:
        state = container_state_fn(spec)
        if state in {"dead", "exited"}:
            raise RuntimeError(f"{spec.role} container state is {state}: {spec.container_name}")
        if state == "running":
            try:
                probe = preflight_fn(spec.base_url, spec.model)
                return {
                    "role": spec.role,
                    "state": state,
                    "probe": probe,
                    "action": "started",
                }
            except Exception as exc:
                last_error = str(exc)
        if monotonic_fn() >= deadline:
            raise RuntimeError(
                f"timed out waiting for {spec.role} service after {timeout}s: {last_error}"
            )
        sleep_fn(2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("print-command", "status", "start"))
    parser.add_argument(
        "target", nargs="?", default="all", choices=("all", "extractor", "judge")
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    specs = selected_services(args.target)
    if args.action == "print-command":
        for spec in specs:
            print(print_command(spec))
        return 0

    results = []
    for spec in specs:
        if args.action == "status":
            results.append(check_service(spec))
        else:
            results.append(start_service(spec, timeout=args.timeout))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
