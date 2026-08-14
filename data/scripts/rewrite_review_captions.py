#!/usr/bin/env python3
"""Align review captions with authoritative crop attributes through Qwen."""

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set


PROMPT_VERSION = "review-caption-attribute-alignment-v1"
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_DIR / "data" / "review"
DEFAULT_WORK_ROOT = PROJECT_DIR / "data" / ".review_caption_rewrite_work"
IGNORED_ATTRIBUTE_VALUES = {
    "",
    "unknown",
    "other",
    "none",
    "no",
    "null",
    "n/a",
    "na",
    "not applicable",
    "not_applicable",
    "unspecified",
}

SYSTEM_PROMPT = """You are a conservative English caption editor.

For every task, compare original_caption with authoritative_attributes.
The attributes provided are meaningful and authoritative because unknown,
absent, and placeholder values have already been removed.

Rules:
1. Correct only facts in the caption that conflict with the authoritative attributes.
2. Naturally add every authoritative attribute that the caption does not mention.
3. Preserve all unrelated visual facts and keep the original wording and sentence structure as much as possible. Restrained synonym changes are allowed only when needed for fluent English.
4. You must not use negative statements or describe anything as absent, missing, unknown, not visible, or not carried.
5. Do not infer any fact that is not in the original caption or authoritative attributes.
6. Write one concise natural English caption with no Markdown or commentary.
7. If a task includes previous_validation_error and retry_instruction, follow that
   retry_instruction carefully.

Return exactly one JSON object in this form:
{"rewrites":[{"id":"the supplied id","caption":"the revised caption"}]}
Include every supplied id exactly once.
"""

RETRY_INSTRUCTION = (
    "Preserve original visual details that do not conflict with "
    "authoritative_attributes. Do not shorten by removing neutral details from "
    "original_caption."
)

_NEGATIVE_PATTERN = re.compile(
    r"\b(?:no|not|without|neither|nor|lack|lacks|lacking|absent|missing|"
    r"unseen|unknown|isn't|aren't|doesn't|don't|hasn't|haven't)\b|"
    r"\b(?:is|are|does|do|has|have)\s+not\b",
    re.IGNORECASE,
)


class ResponseError(RuntimeError):
    """Raised when a model response cannot be mapped safely to input tasks."""


class DatasetError(RuntimeError):
    """Raised when input data cannot be processed safely."""


@dataclass(frozen=True)
class CaptionTask:
    task_id: str
    relative_path: Path
    crop_position: int
    crop_index: Any
    original_caption: str
    attributes: Dict[str, Any]
    original_caption_sha256: str
    attributes_sha256: str

    def as_request(self) -> Dict[str, Any]:
        return {
            "id": self.task_id,
            "original_caption": self.original_caption,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class CachedRewrite:
    caption: str
    applied: bool


@dataclass(frozen=True)
class RequestConfig:
    base_url: str = "http://127.0.0.1:6097/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen36-35b-caption"
    timeout: float = 180.0
    retries: int = 2
    retry_sleep: float = 3.0
    temperature: float = 0.2
    top_p: float = 0.8
    max_tokens: int = 4096


@dataclass(frozen=True)
class PipelineConfig:
    input_root: Path = DEFAULT_INPUT_ROOT
    work_root: Path = DEFAULT_WORK_ROOT
    request: RequestConfig = RequestConfig()
    batch_size: int = 16
    workers: int = 4
    limit_crops: Optional[int] = None


@dataclass(frozen=True)
class BatchResult:
    successes: Dict[str, str]
    failures: Dict[str, str]


@dataclass(frozen=True)
class AuditStats:
    json_count: int
    crop_count: int
    caption_count: int
    meaningful_attribute_crops: int


@dataclass(frozen=True)
class RunStats:
    json_count: int
    total_crops: int
    meaningful_attribute_crops: int
    skipped_no_attributes: int
    cached_applied: int
    cached_pending: int
    requested: int
    updated: int
    failed: int
    limited: bool


_DROP = object()


def _filter_attribute_value(value: Any) -> Any:
    if value is None or value is False:
        return _DROP
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in IGNORED_ATTRIBUTE_VALUES:
            return _DROP
        return normalized
    if isinstance(value, dict):
        filtered: Dict[str, Any] = {}
        for key, child in value.items():
            if str(key).strip().lower() == "extra":
                continue
            kept = _filter_attribute_value(child)
            if kept is not _DROP:
                filtered[str(key)] = kept
        return filtered if filtered else _DROP
    if isinstance(value, list):
        filtered_items = []
        for item in value:
            kept = _filter_attribute_value(item)
            if kept is not _DROP:
                filtered_items.append(kept)
        return filtered_items if filtered_items else _DROP
    return value


def filter_meaningful_attributes(attributes: Any) -> Dict[str, Any]:
    """Return authoritative attributes without placeholders or `extra`."""

    if not isinstance(attributes, dict):
        return {}
    filtered = _filter_attribute_value(attributes)
    return filtered if isinstance(filtered, dict) else {}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attributes_sha256(attributes: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        attributes, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return _sha256_text(serialized)


def _task_id(relative_path: Path, crop_position: int, crop_index: Any) -> str:
    return (
        f"{relative_path.as_posix()}#position={crop_position}"
        f"#crop_index={crop_index}"
    )


def build_caption_task(
    relative_path: Path, crop_position: int, crop: Mapping[str, Any]
) -> CaptionTask:
    caption = crop.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(
            f"missing caption: {relative_path.as_posix()} crop={crop_position}"
        )
    attributes = filter_meaningful_attributes(crop.get("attributes"))
    crop_index = crop.get("crop_index", crop_position + 1)
    normalized_caption = caption.strip()
    return CaptionTask(
        task_id=_task_id(relative_path, crop_position, crop_index),
        relative_path=relative_path,
        crop_position=crop_position,
        crop_index=crop_index,
        original_caption=normalized_caption,
        attributes=attributes,
        original_caption_sha256=_sha256_text(normalized_caption),
        attributes_sha256=_attributes_sha256(attributes),
    )


def _cache_path(work_root: Path, task_id: str) -> Path:
    return work_root / "results" / f"{_sha256_text(task_id)}.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any], mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def save_cached_rewrite(
    work_root: Path,
    task: CaptionTask,
    rewritten_caption: str,
    *,
    model: str,
    prompt_version: str = PROMPT_VERSION,
) -> None:
    error = validate_rewrite(task.original_caption, rewritten_caption)
    if error is not None:
        raise ValueError(f"invalid rewrite for {task.task_id}: {error}")
    payload = {
        "schema_version": "review-caption-rewrite-cache-v1",
        "prompt_version": prompt_version,
        "model": model,
        "task_id": task.task_id,
        "source_relative_path": task.relative_path.as_posix(),
        "crop_position": task.crop_position,
        "crop_index": task.crop_index,
        "original_caption": task.original_caption,
        "original_caption_sha256": task.original_caption_sha256,
        "attributes_sha256": task.attributes_sha256,
        "rewritten_caption": rewritten_caption.strip(),
    }
    _atomic_write_json(_cache_path(work_root, task.task_id), payload)


def load_cached_rewrite(
    work_root: Path,
    relative_path: Path,
    crop_position: int,
    crop: Mapping[str, Any],
    *,
    model: str,
    prompt_version: str = PROMPT_VERSION,
) -> Optional[CachedRewrite]:
    crop_index = crop.get("crop_index", crop_position + 1)
    task_id = _task_id(relative_path, crop_position, crop_index)
    path = _cache_path(work_root, task_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    attributes = filter_meaningful_attributes(crop.get("attributes"))
    if (
        payload.get("schema_version") != "review-caption-rewrite-cache-v1"
        or payload.get("prompt_version") != prompt_version
        or payload.get("model") != model
        or payload.get("task_id") != task_id
        or payload.get("attributes_sha256") != _attributes_sha256(attributes)
    ):
        return None
    current_caption = crop.get("caption")
    original_caption = payload.get("original_caption")
    rewritten_caption = payload.get("rewritten_caption")
    if not isinstance(current_caption, str) or not isinstance(
        rewritten_caption, str
    ):
        return None
    if current_caption.strip() == rewritten_caption:
        return CachedRewrite(rewritten_caption, applied=True)
    if isinstance(original_caption, str) and current_caption.strip() == original_caption:
        return CachedRewrite(rewritten_caption, applied=False)
    return None


def materialize_rewrites(
    input_root: Path,
    tasks: Mapping[str, CaptionTask],
    rewrites: Mapping[str, str],
) -> Dict[str, str]:
    failures: Dict[str, str] = {}
    by_path: Dict[Path, List[str]] = {}
    for task_id in rewrites:
        task = tasks.get(task_id)
        if task is None:
            failures[task_id] = "rewrite has no matching task"
            continue
        by_path.setdefault(task.relative_path, []).append(task_id)

    for relative_path, task_ids in sorted(by_path.items(), key=lambda item: str(item[0])):
        source_path = input_root / relative_path
        try:
            mode = stat.S_IMODE(source_path.stat().st_mode)
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            for task_id in task_ids:
                failures[task_id] = f"cannot read source JSON: {error}"
            continue
        crops = payload.get("crops")
        if not isinstance(crops, list):
            for task_id in task_ids:
                failures[task_id] = "source JSON has no crops list"
            continue

        changed = False
        for task_id in task_ids:
            task = tasks[task_id]
            if task.crop_position >= len(crops) or not isinstance(
                crops[task.crop_position], dict
            ):
                failures[task_id] = "crop position no longer exists"
                continue
            crop = crops[task.crop_position]
            if crop.get("crop_index", task.crop_position + 1) != task.crop_index:
                failures[task_id] = "crop index changed before publication"
                continue
            current_caption = crop.get("caption")
            rewritten_caption = rewrites[task_id].strip()
            if not isinstance(current_caption, str):
                failures[task_id] = "current caption is missing"
                continue
            if current_caption.strip() == rewritten_caption:
                continue
            if current_caption.strip() != task.original_caption:
                failures[task_id] = "caption changed before publication"
                continue
            crop["caption"] = rewritten_caption
            changed = True

        if changed:
            try:
                _atomic_write_json(source_path, payload, mode=mode)
            except OSError as error:
                for task_id in task_ids:
                    if task_id not in failures:
                        failures[task_id] = f"cannot publish source JSON: {error}"
    return failures


def build_chat_payload(
    tasks: Iterable[Dict[str, Any]],
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    retry_feedback: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    request_tasks: List[Dict[str, Any]] = []
    for task in tasks:
        request_task = {
            "id": str(task["id"]),
            "original_caption": str(task["original_caption"]),
            "authoritative_attributes": task["attributes"],
        }
        previous_error = (retry_feedback or {}).get(request_task["id"])
        if previous_error:
            request_task["previous_validation_error"] = previous_error
            request_task["retry_instruction"] = RETRY_INSTRUCTION
        request_tasks.append(request_task)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"tasks": request_tasks}, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


def parse_rewrites(raw: str, expected_ids: Set[str]) -> Dict[str, str]:
    if not isinstance(raw, str) or not raw.strip():
        raise ResponseError("response content is empty")
    if "```" in raw:
        raise ResponseError("response contains Markdown fences")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ResponseError(f"response is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("rewrites"), list):
        raise ResponseError("response must contain a rewrites list")

    rewrites: Dict[str, str] = {}
    for item in payload["rewrites"]:
        if not isinstance(item, dict):
            raise ResponseError("rewrite entry is not an object")
        task_id = item.get("id")
        caption = item.get("caption")
        if not isinstance(task_id, str) or task_id not in expected_ids:
            raise ResponseError(f"unexpected rewrite id: {task_id!r}")
        if task_id in rewrites:
            raise ResponseError(f"duplicate rewrite id: {task_id}")
        if not isinstance(caption, str) or not caption.strip():
            raise ResponseError(f"empty caption for id: {task_id}")
        rewrites[task_id] = caption.strip()

    if set(rewrites) != expected_ids:
        missing = sorted(expected_ids - set(rewrites))
        raise ResponseError(f"missing rewrite ids: {missing}")
    return rewrites


def validate_rewrite(original: str, rewritten: str) -> Optional[str]:
    if not isinstance(rewritten, str) or not rewritten.strip():
        return "caption is empty"
    caption = rewritten.strip()
    if "```" in caption or caption.startswith("#"):
        return "caption contains Markdown"
    if not re.search(r"[A-Za-z]", caption) or re.search(
        r"[\u3400-\u4dbf\u4e00-\u9fff]", caption
    ):
        return "caption is not English text"
    if _NEGATIVE_PATTERN.search(caption):
        return "caption contains negative phrasing"

    original_words = len(str(original).split())
    rewritten_words = len(caption.split())
    if original_words >= 8 and rewritten_words < max(4, int(original_words * 0.5)):
        return "caption is implausibly short"
    if rewritten_words > max(100, original_words * 4 + 24):
        return "caption is implausibly long"
    return None


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base URL must not be empty")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def request_chat(payload: Mapping[str, Any], config: RequestConfig) -> str:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        _chat_completions_url(config.base_url),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise ResponseError(f"HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ResponseError(f"request failed: {error}") from error

    try:
        response_payload = json.loads(raw_response)
        content = response_payload["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise ResponseError("invalid OpenAI-compatible response envelope") from error
    if not isinstance(content, str) or not content.strip():
        raise ResponseError("assistant response content is empty")
    return content


def _request_tasks_once(
    tasks: Sequence[CaptionTask],
    config: RequestConfig,
    requester: Callable[[Mapping[str, Any], RequestConfig], str],
    retry_feedback: Optional[Mapping[str, str]] = None,
) -> BatchResult:
    payload = build_chat_payload(
        [task.as_request() for task in tasks],
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        retry_feedback=retry_feedback,
    )
    expected_ids = {task.task_id for task in tasks}
    raw = requester(payload, config)
    parsed = parse_rewrites(raw, expected_ids)
    successes: Dict[str, str] = {}
    failures: Dict[str, str] = {}
    by_id = {task.task_id: task for task in tasks}
    for task_id, rewritten in parsed.items():
        error = validate_rewrite(by_id[task_id].original_caption, rewritten)
        if error is None:
            successes[task_id] = rewritten
        else:
            failures[task_id] = error
    return BatchResult(successes, failures)


def _retry_tasks(
    tasks: Sequence[CaptionTask],
    config: RequestConfig,
    requester: Callable[[Mapping[str, Any], RequestConfig], str],
) -> BatchResult:
    unresolved = list(tasks)
    successes: Dict[str, str] = {}
    failures: Dict[str, str] = {}
    retry_feedback: Dict[str, str] = {}
    for attempt in range(config.retries + 1):
        if not unresolved:
            break
        try:
            result = _request_tasks_once(
                unresolved,
                config,
                requester,
                retry_feedback=retry_feedback,
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            failures = {task.task_id: message for task in unresolved}
        else:
            successes.update(result.successes)
            failures = dict(result.failures)
            unresolved = [
                task for task in unresolved if task.task_id not in result.successes
            ]
        retry_feedback = {
            task.task_id: failures.get(task.task_id, "rewrite failed")
            for task in unresolved
        }
        if unresolved and attempt < config.retries and config.retry_sleep > 0:
            time.sleep(config.retry_sleep * (2**attempt))
    return BatchResult(successes, failures)


def rewrite_batch(
    tasks: Sequence[CaptionTask],
    config: RequestConfig,
    *,
    requester: Callable[[Mapping[str, Any], RequestConfig], str] = request_chat,
) -> BatchResult:
    if not tasks:
        return BatchResult({}, {})
    batch_result = _retry_tasks(tasks, config, requester)
    successes = dict(batch_result.successes)
    failures = dict(batch_result.failures)
    unresolved = [task for task in tasks if task.task_id not in successes]
    if len(tasks) > 1:
        for task in unresolved:
            individual = _retry_tasks([task], config, requester)
            if task.task_id in individual.successes:
                successes[task.task_id] = individual.successes[task.task_id]
                failures.pop(task.task_id, None)
            else:
                failures[task.task_id] = individual.failures.get(
                    task.task_id, "rewrite failed"
                )
    return BatchResult(successes, failures)


def iter_json_paths(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.json"), key=lambda path: path.as_posix())


def audit_input(root: Path) -> AuditStats:
    root = root.expanduser().resolve(strict=True)
    json_count = 0
    crop_count = 0
    caption_count = 0
    meaningful_attribute_crops = 0
    for path in iter_json_paths(root):
        json_count += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read {path}: {error}") from error
        crops = payload.get("crops")
        if not isinstance(crops, list):
            raise DatasetError(f"missing crops list: {path}")
        crop_count += len(crops)
        for position, crop in enumerate(crops):
            if not isinstance(crop, dict):
                raise DatasetError(f"invalid crop: {path} position={position}")
            caption = crop.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise DatasetError(f"missing caption: {path} position={position}")
            caption_count += 1
            if filter_meaningful_attributes(crop.get("attributes")):
                meaningful_attribute_crops += 1
    return AuditStats(
        json_count,
        crop_count,
        caption_count,
        meaningful_attribute_crops,
    )


def _chunks(items: Sequence[CaptionTask], size: int) -> Iterable[List[CaptionTask]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _write_failures(work_root: Path, failures: Mapping[str, str]) -> None:
    _atomic_write_json(
        work_root / "last_failures.json",
        {
            "schema_version": "review-caption-rewrite-failures-v1",
            "prompt_version": PROMPT_VERSION,
            "failure_count": len(failures),
            "failures": dict(sorted(failures.items())),
        },
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    requester: Callable[[Mapping[str, Any], RequestConfig], str] = request_chat,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> RunStats:
    input_root = config.input_root.expanduser().resolve(strict=True)
    work_root = config.work_root.expanduser().absolute()
    if config.batch_size < 1 or config.workers < 1:
        raise ValueError("batch size and workers must be positive")
    if config.limit_crops is not None and config.limit_crops < 1:
        raise ValueError("limit crops must be positive")

    json_count = 0
    total_crops = 0
    meaningful_count = 0
    skipped_no_attributes = 0
    cached_applied = 0
    cached_pending = 0
    scan_failures: Dict[str, str] = {}
    cached_tasks: Dict[str, CaptionTask] = {}
    cached_rewrites: Dict[str, str] = {}
    pending_tasks: List[CaptionTask] = []

    for path in iter_json_paths(input_root):
        json_count += 1
        relative_path = path.relative_to(input_root)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            scan_failures[relative_path.as_posix()] = f"cannot read JSON: {error}"
            continue
        crops = payload.get("crops")
        if not isinstance(crops, list):
            scan_failures[relative_path.as_posix()] = "missing crops list"
            continue
        total_crops += len(crops)
        for position, crop in enumerate(crops):
            fallback_id = f"{relative_path.as_posix()}#position={position}"
            if not isinstance(crop, dict):
                scan_failures[fallback_id] = "crop is not an object"
                continue
            attributes = filter_meaningful_attributes(crop.get("attributes"))
            if not attributes:
                skipped_no_attributes += 1
                continue
            meaningful_count += 1
            try:
                task = build_caption_task(relative_path, position, crop)
            except ValueError as error:
                scan_failures[fallback_id] = str(error)
                continue
            cached = load_cached_rewrite(
                work_root,
                relative_path,
                position,
                crop,
                model=config.request.model,
            )
            if cached is not None and cached.applied:
                cached_applied += 1
                continue
            if cached is not None:
                cached_pending += 1
                cached_tasks[task.task_id] = task
                cached_rewrites[task.task_id] = cached.caption
            else:
                pending_tasks.append(task)

    limited = False
    if config.limit_crops is not None and len(pending_tasks) > config.limit_crops:
        pending_tasks = pending_tasks[: config.limit_crops]
        limited = True

    task_map: Dict[str, CaptionTask] = dict(cached_tasks)
    task_map.update({task.task_id: task for task in pending_tasks})
    successful_rewrites = dict(cached_rewrites)
    request_failures: Dict[str, str] = {}
    requested = len(pending_tasks)
    batches = list(_chunks(pending_tasks, config.batch_size))

    if batches:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            future_to_batch = {
                executor.submit(
                    rewrite_batch,
                    batch,
                    config.request,
                    requester=requester,
                ): batch
                for batch in batches
            }
            completed = 0
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    result = BatchResult(
                        {}, {task.task_id: message for task in batch}
                    )
                for task_id, caption in result.successes.items():
                    task = task_map[task_id]
                    try:
                        save_cached_rewrite(
                            work_root,
                            task,
                            caption,
                            model=config.request.model,
                        )
                    except (OSError, ValueError) as error:
                        request_failures[task_id] = f"cannot cache result: {error}"
                    else:
                        successful_rewrites[task_id] = caption
                request_failures.update(result.failures)
                completed += len(batch)
                if progress is not None:
                    progress(
                        {
                            "event": "batch_done",
                            "completed": completed,
                            "requested": requested,
                            "successes": len(result.successes),
                            "failures": len(result.failures),
                        }
                    )

    publication_failures = materialize_rewrites(
        input_root, task_map, successful_rewrites
    )
    all_failures = dict(scan_failures)
    all_failures.update(request_failures)
    all_failures.update(publication_failures)
    _write_failures(work_root, all_failures)
    updated = len(successful_rewrites) - len(
        set(successful_rewrites) & set(publication_failures)
    )
    return RunStats(
        json_count=json_count,
        total_crops=total_crops,
        meaningful_attribute_crops=meaningful_count,
        skipped_no_attributes=skipped_no_attributes,
        cached_applied=cached_applied,
        cached_pending=cached_pending,
        requested=requested,
        updated=updated,
        failed=len(all_failures),
        limited=limited,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align data/review crop captions with meaningful attributes through "
            "the text-only Qwen vLLM service, updating captions in place."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:6097/v1"),
    )
    parser.add_argument(
        "--model", default=os.environ.get("QWEN_MODEL", "Qwen36-35b-caption")
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("QWEN_API_KEY", "EMPTY")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--limit-crops",
        type=int,
        default=None,
        help="Process at most this many uncached crops in this run.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate and count input data without calling Qwen or modifying JSON.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        audit = audit_input(args.input_root)
    except (DatasetError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"input json={audit.json_count} crops={audit.crop_count} "
        f"captions={audit.caption_count} "
        f"meaningful_attributes={audit.meaningful_attribute_crops}",
        flush=True,
    )
    if args.audit_only:
        return 0

    request_config = RequestConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    pipeline_config = PipelineConfig(
        input_root=args.input_root,
        work_root=args.work_root,
        request=request_config,
        batch_size=args.batch_size,
        workers=args.workers,
        limit_crops=args.limit_crops,
    )

    def print_progress(event: Dict[str, Any]) -> None:
        print(
            f"batch completed={event['completed']}/{event['requested']} "
            f"ok={event['successes']} failed={event['failures']}",
            flush=True,
        )

    try:
        stats = run_pipeline(pipeline_config, progress=print_progress)
    except (DatasetError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"done json={stats.json_count} crops={stats.total_crops} "
        f"meaningful={stats.meaningful_attribute_crops} "
        f"skipped_no_attributes={stats.skipped_no_attributes} "
        f"cached_applied={stats.cached_applied} "
        f"cached_pending={stats.cached_pending} requested={stats.requested} "
        f"updated={stats.updated} failed={stats.failed} limited={stats.limited}",
        flush=True,
    )
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
