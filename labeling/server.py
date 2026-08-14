#!/usr/bin/env python3
"""
标注审核服务器：
  - 提供静态文件服务（annotation_review.html）
  - POST /save  接收标注结果 JSON 并写入 review_results/ 目录
用法：python3 server.py [端口]  默认 8888
"""

import json
import os
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

SAVE_DIR = Path(__file__).parent / "review_results"
SAVE_DIR.mkdir(exist_ok=True)

SERVE_DIR = Path(__file__).parent


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_DIR), **kwargs)

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": f"JSON 解析失败: {e}"})
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"review_{ts}.json"
        save_path = SAVE_DIR / filename

        try:
            save_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self._json_response(500, {"error": f"写入失败: {e}"})
            return

        print(f"[SAVED] {save_path}")
        self._json_response(200, {"path": str(save_path), "filename": filename})

    def _json_response(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        # 过滤掉图片请求的日志，只打印 POST
        if args and str(args[0]).startswith("POST"):
            super().log_message(fmt, *args)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="标注审核服务器")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8990,
        help="监听端口（默认 8990）"
    )
    args = parser.parse_args()
    port = args.port

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"服务器启动：http://0.0.0.0:{port}")
    print(f"静态文件目录：{SERVE_DIR}")
    print(f"结果保存目录：{SAVE_DIR}")
    print(f"\n在 Edge 中访问：http://192.168.111.2:{port}/annotation_review.html")
    print("按 Ctrl+C 停止\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")


if __name__ == "__main__":
    main()
