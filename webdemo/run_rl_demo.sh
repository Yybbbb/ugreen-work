#!/usr/bin/env bash
# Run the RL comparison demo (build if needed, then serve on port 8013)
set -e
cd "$(dirname "$0")/.."

OUT_DIR="webdemo/rl_build"
if [[ ! -f "$OUT_DIR/index.html" ]]; then
    echo "Building RL demo …"
    python webdemo/build_rl_demo.py
fi

echo "Starting RL demo server on :8013 …"
setsid python -m uvicorn "webdemo.rl_server:create_app" \
    --factory --host 0.0.0.0 --port 8013 \
    > /tmp/rl_demo.log 2>&1 < /dev/null &
echo "PID=$! · http://localhost:8013"
