#!/usr/bin/env python3
"""
调用 Qwen3.6-35B 服务，判断 object_detection_0309-0429 中每张照片是否包含
较完整的人物（头 + 上半身），结果写入 person_check_results.json。

特性:
  - 断点续跑：已有结果自动跳过
  - 并发请求：默认 8 线程
  - 增量保存：每 100 张写一次 JSON
  - 失败自动重试：网络/服务错误最多重试 3 次
"""

import base64
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

# -------- 配置 --------
ROOT_DIR = Path(__file__).resolve().parent.parent / "object_detection_0309-0429"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "person_check_prompt.txt"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "person_check_results.json"
API_URL = "http://localhost:6096/v1/chat/completions"
MODEL_NAME = "Qwen36-35b"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
CONCURRENCY = 8          # 并发线程数
SAVE_INTERVAL = 100      # 每处理 N 张保存一次
MAX_RETRIES = 3          # 单张失败重试次数
RETRY_DELAY = 3          # 重试间隔（秒）
MAX_TOKENS = 2048        # 推理模型需要足够 token 完成思考 + 输出 JSON


# -------- 工具函数 --------

def load_prompt(path: Path) -> str:
    """从 txt 文件读取 prompt。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def scan_images(root: Path) -> list:
    """扫描所有图片，返回相对路径列表（按文件名排序，保证可复现）。"""
    images = []
    for fp in sorted(root.rglob("*")):
        if fp.suffix.lower() in IMAGE_EXTS:
            images.append(fp)
    return images


def encode_image_base64(filepath: Path) -> str:
    """将图片编码为 base64 data URL。"""
    # 根据扩展名确定 MIME 类型
    ext = filepath.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def parse_response(text: str) -> Optional[bool]:
    """从模型返回的文本中提取 has_person 布尔值。"""
    # 去掉可能的 markdown 代码块标记
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ``` 或 ``` ... ```
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    # 尝试多种 JSON 提取方式
    candidates = [text]

    # 找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if "has_person" in obj:
                return bool(obj["has_person"])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # 最后兜底：直接搜索 true/false
    if '"has_person": true' in text.lower() or '"has_person":true' in text.lower():
        return True
    if '"has_person": false' in text.lower() or '"has_person":false' in text.lower():
        return False

    return None


# -------- 单张图片判断 --------

def check_single_image(
    prompt: str,
    filepath: Path,
    rel_path: str,
    session: requests.Session,
) -> Tuple[str, Optional[bool], Optional[str]]:
    """
    对单张图片调用模型判断。
    返回 (rel_path, result, error_message)
      - result=True/False: 正常判断
      - result=None, error_message 非空: 失败
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
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": b64_url}},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(API_URL, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                result = parse_response(content)
                if result is not None:
                    return (rel_path, result, None)
                else:
                    # 解析失败也重试
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

    # 2. 加载 prompt
    if not PROMPT_FILE.exists():
        print(f"[ERROR] Prompt 文件不存在: {PROMPT_FILE}")
        sys.exit(1)
    prompt = load_prompt(PROMPT_FILE)
    print(f"已加载 prompt ({len(prompt)} 字符): {PROMPT_FILE}\n")

    # 3. 扫描图片
    all_images = scan_images(ROOT_DIR)
    total = len(all_images)
    print(f"扫描到 {total:,} 张图片: {ROOT_DIR}\n")

    # 构建 相对路径 → 绝对路径 映射
    image_map = {}
    for fp in all_images:
        rel = str(fp.relative_to(ROOT_DIR))
        image_map[rel] = fp

    # 4. 加载已有结果（断点续跑）
    results: Dict[str, bool] = {}
    errors: Dict[str, str] = {}   # rel_path → error_message
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            # 兼容两种格式
            if isinstance(existing, dict) and "results" in existing:
                results = existing["results"]
                if "errors" in existing:
                    errors = existing["errors"]
            else:
                # 纯 dict 格式: {rel_path: bool, ...}
                for k, v in existing.items():
                    if isinstance(v, bool):
                        results[k] = v
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
        return check_single_image(prompt, abs_path, rel, session)

    print(f"开始处理 ({CONCURRENCY} 线程)...")
    print(f"进度             耗时    结果(H/N/E)  最新文件")
    print("-" * 75)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process_one, p): p for p in pending}

        for future in as_completed(futures):
            rel, result, error = future.result()
            with lock:
                if result is not None:
                    results[rel] = result
                else:
                    errors[rel] = error or "unknown"
                processed_count += 1

                # 增量保存
                if processed_count % SAVE_INTERVAL == 0 or processed_count == len(pending):
                    has_person = sum(1 for v in results.values() if v)
                    no_person = sum(1 for v in results.values() if not v)
                    elapsed = time.time() - start_time
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - processed_count) / speed if speed > 0 else 0

                    output = {
                        "meta": {
                            "prompt_file": str(PROMPT_FILE.name),
                            "model": MODEL_NAME,
                            "endpoint": API_URL,
                            "total_images": total,
                            "processed": len(results) + len(errors),
                            "has_person": has_person,
                            "no_person": no_person,
                            "error": len(errors),
                        },
                        "results": dict(sorted(results.items())),
                        "errors": dict(sorted(errors.items())),
                    }
                    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                        json.dump(output, f, ensure_ascii=False, indent=2)

                    hms = f"{int(elapsed//3600):02d}:{int(elapsed%3600//60):02d}:{int(elapsed%60):02d}"
                    print(f"  {len(results)+len(errors):>5,}/{total:,}"
                          f"  {hms}"
                          f"  {has_person}/{no_person}/{len(errors)}"
                          f"  {rel}")

    # 6. 最终统计
    has_person = sum(1 for v in results.values() if v)
    no_person = sum(1 for v in results.values() if not v)
    elapsed = time.time() - start_time
    hms = f"{int(elapsed//3600):02d}:{int(elapsed%3600//60):02d}:{int(elapsed%60):02d}"

    print()
    print("=" * 55)
    print("  处理完成!")
    print(f"  总耗时:    {hms}")
    print(f"  已处理:    {len(results) + len(errors):,} / {total:,}")
    print(f"  有人物:    {has_person:,}")
    print(f"  无人物:    {no_person:,}")
    print(f"  失败:      {len(errors):,}")
    print(f"  结果文件:  {OUTPUT_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    main()
