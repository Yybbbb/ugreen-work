#!/usr/bin/env python3
"""
调用 Qwen3.6-35B 服务，对 object_detection_0309-0429 中有人物的照片
进行衣服颜色标注，结果写入 qwen_color_clean_results.json。

特性:
  - 仅处理 person_check_results.json 中 has_person=true 的图片
  - 断点续跑：已有结果自动跳过
  - 并发请求：默认 8 线程
  - 增量保存：每 100 张写一次 JSON
  - 失败自动重试：网络/服务错误最多重试 3 次

颜色标签（12 选 1，含 unknown）:
  black, white, gray, blue, red, green, brown,
  yellow, pink, purple, orange, unknown

unknown 触发条件：
  - 图片模糊、过暗、过曝无法判断颜色
  - 衣服有多色拼接（color-block）且无主导色
  - 有条纹、格子、千鸟格等多色图案
  - 多种颜色无法确定哪一个占主导
"""

import base64
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

# -------- 配置 --------
ROOT_DIR = Path(__file__).resolve().parent.parent / "object_detection_0309-0429"
PERSON_CHECK = Path(__file__).resolve().parent.parent / "person_check_results.json"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "qwen_color_clean_results.json"
API_URL = "http://localhost:6096/v1/chat/completions"
MODEL_NAME = "Qwen36-35b"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
CONCURRENCY = 8
SAVE_INTERVAL = 100
MAX_RETRIES = 3
RETRY_DELAY = 3
MAX_TOKENS = 1024

ALLOWED_LABELS = {
    "black", "white", "gray", "blue", "red", "green",
    "brown", "yellow", "pink", "purple", "orange", "unknown",
}

# -------- Prompt --------
PROMPT = """/no_think

You are a clothing color annotation assistant. Output ONLY a single valid JSON object. No thinking, no explanation, no markdown.

## Task
Look at the person in this image. Identify the dominant garment color for BOTH the upper-body garment (upper) and the lower-body garment (lower). You MUST output BOTH "upper" and "lower" keys — this is NON-NEGOTIABLE.

## IMPORTANT — Analysis process (follow this order for EVERY image):
### Step 1: Locate upper and lower body
First, carefully examine the person's pose, orientation, and framing in the image:
- Identify which part of the body is the UPPER body (chest, torso, shoulders, arms) and which part is the LOWER body (waist down, pants, skirt, legs).
- Pay attention to the person's posture: are they sitting, standing, bending, or partially cropped? This affects where upper/lower boundaries are.
- If the image is a close-up (e.g., half-body shot), the lower body may be invisible — mark it as "unknown".
- If the image shows the full body but the person is at an unusual angle, think carefully about which clothing region is upper vs lower.

### Step 2: Consider environment and lighting
Before deciding the color, assess the environmental and lighting conditions that may affect color perception:
- **Lighting**: Is the scene indoors or outdoors? Bright sunlight, shadow, warm indoor light, cool fluorescent light, or dim/dark environment? Lighting can shift how a color appears (e.g., white under warm light may look yellowish; dark blue in shadow may look black).
- **Shadows**: Are parts of the garment in shadow? A shadowed area of a white shirt may look gray — judge the well-lit portion.
- **Reflections / glare**: Is there strong backlight or glare washing out the color?
- **Image quality**: Is the image sharp or blurry? Low resolution or motion blur may make color boundaries fuzzy.
- After identifying these factors, mentally compensate for them and determine the TRUE dominant garment color, not the apparent color affected by lighting.

## Allowed color labels (ONLY these 12 — no other values):
black, white, gray, blue, red, green, brown, yellow, pink, purple, orange, unknown

## CRITICAL — When to choose "unknown" for a part:
Choose "unknown" when ANY of the following is true for that garment part:
- The part is fully occluded, cut off, or completely invisible in the image
- The image is too blurry, dark, or overexposed to reliably judge the color of that part
- The garment has multiple distinct colors stitched/joined together (color-block design) with NO single dominant color
- The garment has stripes, plaid/check patterns, or other multi-color geometric patterns
- The garment has multi-color floral prints or complex patterns with no clear dominant color
- You genuinely cannot decide between two or more color labels for that part
- The person is too far away or too small to discern clothing color

## When to choose a specific color:
- The garment is essentially one overall color (ignore shadows, wrinkles, lighting, small logos, buttons, zippers, trim)
- The garment has a clear dominant background color even with minor decorative elements
- The color is clearly identifiable despite poor lighting

## Rules
- ALWAYS output BOTH "upper" and "lower" keys, no exceptions.
- Output exactly ONE color per part (single label + confidence).
- Do NOT output multiple colors for a single part — use "unknown" instead when there are multiple colors.
- Confidence is a float from 0.0 to 1.0.
- Confidence should be >= 0.60 when outputting a specific color.
- When confidence would be below 0.60, output "unknown" instead.

## Required output format (JSON only):
{
  "upper": [{"label": "black", "confidence": 0.92}],
  "lower": [{"label": "blue", "confidence": 0.87}]
}

Example with one part occluded:
{
  "upper": [{"label": "white", "confidence": 0.89}],
  "lower": [{"label": "unknown", "confidence": 0.50}]
}

Output ONLY the JSON object — no markdown fences, no commentary."""


# -------- 工具函数 --------

def scan_has_person_images(root: Path, person_check: Path) -> list:
    """读取 person_check_results.json，返回 has_person=true 的图片绝对路径列表。"""
    with open(person_check, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    has_person = [rel for rel, v in results.items() if v]

    # 验证图片存在
    valid = []
    for rel in has_person:
        fp = root / rel
        if fp.exists():
            valid.append(fp)
    return valid


def encode_image_base64(filepath: Path) -> str:
    ext = filepath.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
        ".webp": "image/webp", ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def parse_response(text: str) -> Optional[dict]:
    """从模型返回文本中提取 {"upper": [...], "lower": [...]}。"""
    text = text.strip()

    # 去掉 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
        text = text.strip()

    # 尝试直接解析
    try:
        obj = json.loads(text)
        if "upper" in obj and "lower" in obj:
            return obj
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # 找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if "upper" in obj and "lower" in obj:
                return obj
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return None


def validate_result(parsed: dict) -> dict:
    """确保 upper/lower 都存在，每个 part 取第一个 label，校验合法性。"""
    cleaned = {}
    for part in ("upper", "lower"):
        entries = parsed.get(part, [])
        if not isinstance(entries, list) or len(entries) == 0:
            # 缺失则该 part 强制 unknown
            cleaned[part] = [{"label": "unknown", "confidence": 0.0, "_corrected_from": "missing"}]
            continue

        first = entries[0]
        if not isinstance(first, dict):
            cleaned[part] = [{"label": "unknown", "confidence": 0.0, "_corrected_from": str(first)}]
            continue

        label = first.get("label", "unknown")
        confidence = first.get("confidence", 0.5)
        entry = {"label": label, "confidence": confidence}

        if label not in ALLOWED_LABELS:
            entry["label"] = "unknown"
            entry["_corrected_from"] = label

        cleaned[part] = [entry]
    return cleaned


def check_single_image(
    filepath: Path,
    rel_path: str,
    session: requests.Session,
) -> Tuple[str, Optional[dict], Optional[str]]:
    """
    对单张图片调用模型进行颜色标注。
    返回 (rel_path, parsed_dict, error_message)
    """
    try:
        b64_url = encode_image_base64(filepath)
    except Exception as e:
        return (rel_path, None, f"encode error: {e}")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": b64_url}},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(API_URL, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = parse_response(content)
                if parsed is not None:
                    parsed = validate_result(parsed)
                    return (rel_path, parsed, None)
                else:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    return (rel_path, None, f"parse error: {content[:200]}")
            else:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                    continue
                return (rel_path, None, f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                continue
            return (rel_path, None, "timeout after retries")
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                continue
            return (rel_path, None, "connection error after retries")
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                continue
            return (rel_path, None, f"unknown: {e}")

    return (rel_path, None, "max retries exceeded")


# -------- 主逻辑 --------

def main():
    # 1. 检查服务
    print(f"检查服务: {API_URL}")
    try:
        r = requests.get("http://localhost:6096/health", timeout=5)
        if r.status_code != 200:
            print("[ERROR] 服务不可用，请确认 vllm-qwen36 容器正在运行")
            sys.exit(1)
    except Exception:
        print("[ERROR] 无法连接服务，请确认 vllm-qwen36 容器正在运行")
        sys.exit(1)
    print("  服务 OK\n")

    # 2. 检查 person_check_results.json
    if not PERSON_CHECK.exists():
        print(f"[ERROR] person_check_results.json 不存在: {PERSON_CHECK}")
        sys.exit(1)

    # 3. 加载 has_person 图片
    images = scan_has_person_images(ROOT_DIR, PERSON_CHECK)
    total = len(images)
    print(f"has_person 图片: {total:,} 张\n")

    # 构建 相对路径 → 绝对路径 映射
    image_map = {}
    for fp in images:
        rel = str(fp.relative_to(ROOT_DIR))
        image_map[rel] = fp

    # 4. 加载已有结果（断点续跑）
    results: Dict[str, dict] = {}
    errors: Dict[str, str] = {}
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and "results" in existing:
                results = existing["results"]
                if "errors" in existing:
                    errors = existing["errors"]
            print(f"加载已有结果: {len(results):,} 张 (跳过)\n")
        except Exception as e:
            print(f"[WARN] 读取已有结果失败: {e}，从头开始\n")

    # 构建待处理列表
    pending = [(rel, abs_path) for rel, abs_path in image_map.items()
               if rel not in results and rel not in errors]
    print(f"待处理: {len(pending):,} / 合计: {total:,}\n")

    if not pending:
        print("全部已完成，退出。")
        return

    # 5. 并发处理
    session_local = threading.local()
    lock = threading.Lock()
    processed_count = 0
    start_time = time.time()

    def get_session():
        if not hasattr(session_local, "session"):
            session_local.session = requests.Session()
            session_local.session.headers.update({"Content-Type": "application/json"})
        return session_local.session

    def process_one(item):
        rel, abs_path = item
        session = get_session()
        return check_single_image(abs_path, rel, session)

    print(f"开始处理 ({CONCURRENCY} 线程)...")
    print(f"进度             耗时    H/N/U/E  最新文件")
    print("-" * 75)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process_one, p): p for p in pending}

        for future in as_completed(futures):
            rel, parsed, error = future.result()
            with lock:
                if parsed is not None:
                    results[rel] = parsed
                else:
                    errors[rel] = error or "unknown"
                processed_count += 1

                # 增量保存
                if processed_count % SAVE_INTERVAL == 0 or processed_count == len(pending):
                    # 统计分布 (upper / lower 分开)
                    from collections import Counter
                    upper_counter = Counter()
                    lower_counter = Counter()
                    for v in results.values():
                        upper_counter[v["upper"][0].get("label", "unknown")] += 1
                        lower_counter[v["lower"][0].get("label", "unknown")] += 1

                    elapsed = time.time() - start_time
                    speed = processed_count / elapsed if elapsed > 0 else 0

                    output = {
                        "meta": {
                            "model": MODEL_NAME,
                            "endpoint": API_URL,
                            "total_has_person": total,
                            "processed": len(results) + len(errors),
                            "success": len(results),
                            "error": len(errors),
                            "upper_distribution": dict(upper_counter.most_common()),
                            "lower_distribution": dict(lower_counter.most_common()),
                        },
                        "results": dict(sorted(results.items())),
                        "errors": dict(sorted(errors.items())),
                    }
                    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                        json.dump(output, f, ensure_ascii=False, indent=2)

                    hms = f"{int(elapsed//3600):02d}:{int(elapsed%3600//60):02d}:{int(elapsed%60):02d}"
                    total_done = len(results) + len(errors)
                    print(f"  {total_done:>5,}/{total:,}"
                          f"  {hms}"
                          f"  ok:{len(results)} err:{len(errors)}"
                          f"  {rel}")

    # 6. 最终统计
    from collections import Counter
    upper_counter = Counter()
    lower_counter = Counter()
    confidences = []
    for v in results.values():
        for part in ("upper", "lower"):
            entry = v[part][0]
            lbl = entry.get("label", "unknown")
            conf = entry.get("confidence")
            if isinstance(conf, (int, float)):
                confidences.append(conf)
            if part == "upper":
                upper_counter[lbl] += 1
            else:
                lower_counter[lbl] += 1

    elapsed = time.time() - start_time
    hms = f"{int(elapsed//3600):02d}:{int(elapsed%3600//60):02d}:{int(elapsed%60):02d}"

    total_results = len(results)
    upper_unknown = upper_counter.get("unknown", 0)
    lower_unknown = lower_counter.get("unknown", 0)

    # 最终写入
    output = {
        "meta": {
            "model": MODEL_NAME,
            "endpoint": API_URL,
            "total_has_person": total,
            "processed": len(results) + len(errors),
            "success": len(results),
            "error": len(errors),
            "upper_distribution": dict(upper_counter.most_common()),
            "lower_distribution": dict(lower_counter.most_common()),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "upper_unknown_rate": upper_unknown / total_results if total_results else 0,
            "lower_unknown_rate": lower_unknown / total_results if total_results else 0,
        },
        "results": dict(sorted(results.items())),
        "errors": dict(sorted(errors.items())),
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("  处理完成!")
    print(f"  总耗时:              {hms}")
    print(f"  成功:                {len(results):,}")
    print(f"  失败:                {len(errors):,}")
    print(f"  平均置信度:          {output['meta']['avg_confidence']:.3f}")
    print(f"  Upper unknown 比例:  {output['meta']['upper_unknown_rate']:.1%}")
    print(f"  Lower unknown 比例:  {output['meta']['lower_unknown_rate']:.1%}")
    print(f"  --- Upper 颜色分布 ---")
    for lbl, cnt in upper_counter.most_common():
        bar = "█" * int(cnt / max(upper_counter.values()) * 20) if upper_counter else ""
        print(f"    {lbl:<10} {cnt:>6,}  {bar}")
    print(f"  --- Lower 颜色分布 ---")
    for lbl, cnt in lower_counter.most_common():
        bar = "█" * int(cnt / max(lower_counter.values()) * 20) if lower_counter else ""
        print(f"    {lbl:<10} {cnt:>6,}  {bar}")
    print(f"  结果文件:            {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
