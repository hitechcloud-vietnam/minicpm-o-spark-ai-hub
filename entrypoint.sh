#!/usr/bin/env bash
set -euo pipefail

cd /app/web_demos/minicpm-o_2.6

echo "[minicpm-o] starting model_server on :32550 (model=${MINICPM_MODEL})"
python model_server.py --port 32550 --model "${MINICPM_MODEL}" &
MODEL_PID=$!

# Health-probe the backend, but don't block nginx forever — user will see
# a friendly "loading" state while weights download on first run.
for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:32550/api/v1/ping >/dev/null 2>&1; then
        echo "[minicpm-o] model_server ready"
        break
    fi
    if ! kill -0 "$MODEL_PID" 2>/dev/null; then
        echo "[minicpm-o] model_server exited during startup" >&2
        wait "$MODEL_PID" || true
        exit 1
    fi
    sleep 2
done

echo "[minicpm-o] starting nginx on :7860"
exec nginx -c /etc/nginx/nginx.conf
