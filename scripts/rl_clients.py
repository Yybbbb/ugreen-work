#!/usr/bin/env python3
"""vLLM clients for the RL reward pipeline: attribute extractor + value-match judge.

Import-safe (stdlib only). Both call OpenAI-compatible vLLM chat endpoints with
thinking disabled and temperature 0. The extractor reuses the fixed-schema prompt
from extract_qwen_attributes (unknown for absent fields); the judge uses
JUDGE_PROMPT from rl_reward to decide match vs conflict per attribute value.
"""

import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from extract_qwen_attributes import PROMPT as EXTRACTOR_PROMPT
from rl_reward import JUDGE_PROMPT


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _models_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S).strip()


def parse_judge_content(text: str) -> float:
    """Parse judge output into a similarity score in {1.0, 0.75, 0.5, 0.25, 0.0}.

    Defaults to 0.0 (conflict) on any parse failure: a conservative choice that
    penalizes the caption rather than letting a possible hallucination through.
    """
    cleaned = _strip_fences(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "score" in obj:
            return _snap_score(float(obj["score"]))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # plain-text fallback: scan for one of the 5 levels
    lower = cleaned.lower()
    for token, score in (("1.0", 1.0), ("0.75", 0.75), ("0.5", 0.5), ("0.25", 0.25), ("0.0", 0.0)):
        if token in lower:
            return score
    return 0.0


_SCORE_LEVELS = (1.0, 0.75, 0.5, 0.25, 0.0)


def _snap_score(value: float) -> float:
    """Snap a float to the nearest allowed 5-level score."""
    return min(_SCORE_LEVELS, key=lambda level: abs(level - value))


def parse_extractor_content(text: str) -> List[Dict[str, Any]]:
    """Parse extractor JSON (results wrapper, single object, or list)."""
    cleaned = _strip_fences(text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        return parsed["results"]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def _get_json(url: str, *, timeout: float = 10.0, retries: int = 2) -> Dict[str, Any]:
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"Authorization": "Bearer EMPTY"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            if not isinstance(envelope, dict):
                raise ValueError("response is not a JSON object")
            return envelope
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"HTTP GET failed after {retries + 1} attempts: {last_error}")


def _post_chat_message(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    timeout: float,
    retries: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    response_format: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if response_format is not None:
        payload["response_format"] = response_format
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                _endpoint(base_url), data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            message = envelope["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("response message is not a JSON object")
            return message
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
                ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"vLLM call failed after {retries + 1} attempts: {last_error}")


def _post_chat(*, base_url: str, model: str, messages: List[Dict[str, str]],
               max_tokens: int, timeout: float, retries: int,
               temperature: float = 0.0, top_p: float = 1.0,
               response_format: Optional[Dict[str, str]] = None) -> str:
    message = _post_chat_message(
        base_url=base_url,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        temperature=temperature,
        top_p=top_p,
        response_format=response_format,
    )
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("vLLM response content is not a string")
    return content


def parse_model_ids(envelope: Dict[str, Any]) -> List[str]:
    data = envelope.get("data")
    if not isinstance(data, list):
        return []
    return [
        item["id"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def validate_probe_message(message: Dict[str, Any]) -> None:
    content = message.get("content")
    if not isinstance(content, str) or content.strip() != "OK":
        raise RuntimeError(f"reward service probe expected content 'OK', got {content!r}")
    for key, value in message.items():
        if "reasoning" in key.lower() and value not in (None, "", [], {}):
            raise RuntimeError(f"reward service probe returned unexpected reasoning in {key}")


def preflight_service(
    base_url: str,
    model: str,
    *,
    timeout: float = 10.0,
    retries: int = 2,
) -> Dict[str, str]:
    model_ids = parse_model_ids(
        _get_json(_models_endpoint(base_url), timeout=timeout, retries=retries)
    )
    if model not in model_ids:
        raise RuntimeError(
            f"reward service expected model {model!r}, served models={model_ids!r}"
        )
    message = _post_chat_message(
        base_url=base_url,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return exactly OK without explanation or reasoning.",
            },
            {"role": "user", "content": "Reply with exactly OK."},
        ],
        max_tokens=8,
        timeout=timeout,
        retries=retries,
        temperature=0.0,
        top_p=1.0,
    )
    validate_probe_message(message)
    return {"model": model, "content": str(message["content"]).strip()}


def extract_attributes(captions: Sequence[str], *, base_url: str, model: str,
                       timeout: float = 180.0, retries: int = 2,
                       max_tokens: int = 1024) -> List[Optional[Dict[str, Any]]]:
    """Extract fixed-schema attributes for a batch of captions. None on failure."""
    items = [{"id": str(i), "caption": caption} for i, caption in enumerate(captions)]
    user = EXTRACTOR_PROMPT + "\n\nItems:\n" + json.dumps(items, ensure_ascii=False)
    messages = [
        {"role": "system", "content": "You are a precise person-attribute extraction service. Output JSON only."},
        {"role": "user", "content": user},
    ]
    content = _post_chat(
        base_url=base_url, model=model, messages=messages,
        max_tokens=max_tokens, timeout=timeout, retries=retries,
        temperature=0.0, top_p=1.0, response_format={"type": "json_object"},
    )
    parsed = parse_extractor_content(content)
    by_id = {str(item.get("id")): item for item in parsed
             if isinstance(item, dict) and item.get("id") is not None}
    return [by_id.get(str(i)) for i in range(len(captions))]


def make_judge_fn(base_url: str, model: str, *, timeout: float = 60.0, retries: int = 2,
                  max_tokens: int = 64, concurrency: int = 16):
    """Return a cached similarity_fn(field, gt_value, gen_value) -> float.

    The float is a 5-level semantic similarity {1.0, 0.75, 0.5, 0.25, 0.0} from
    the Qwen judge. Cache is keyed by (field, gt_value, gen_value); the same
    value pair is never re-queried within a process. Thread-safe.
    """
    if concurrency <= 0:
        raise ValueError("judge concurrency must be positive")
    cache: Dict[str, float] = {}
    lock = threading.Lock()

    def similarity_fn(field: str, gt_value: str, gen_value: str) -> float:
        key = f"{field}\x1f{gt_value}\x1f{gen_value}"
        with lock:
            cached = cache.get(key)
        if cached is not None:
            return cached
        prompt = JUDGE_PROMPT.format(field=field, gt_value=gt_value, gen_value=gen_value)
        messages = [
            {"role": "system", "content": "You are a strict attribute-value similarity scorer. Output JSON only."},
            {"role": "user", "content": prompt},
        ]
        content = _post_chat(
            base_url=base_url, model=model, messages=messages,
            max_tokens=max_tokens, timeout=timeout, retries=retries,
            temperature=0.0, top_p=1.0, response_format={"type": "json_object"},
        )
        score = parse_judge_content(content)
        with lock:
            cache[key] = score
        return score

    def prefetch(requests) -> None:
        pending = []
        seen = set()
        with lock:
            cached_keys = set(cache)
        for field, gt_value, gen_value in requests:
            item = (str(field), str(gt_value), str(gen_value))
            key = "\x1f".join(item)
            if key not in cached_keys and key not in seen:
                seen.add(key)
                pending.append(item)
        if not pending:
            return
        with ThreadPoolExecutor(max_workers=min(concurrency, len(pending))) as executor:
            list(executor.map(lambda item: similarity_fn(*item), pending))

    similarity_fn.prefetch = prefetch
    return similarity_fn
