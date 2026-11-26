#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  echo "Error: Python virtual environment (.venv) not found."
  echo "Run: uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e '.[dev]'"
  exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Shutting down VoiceCast..."
  if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting FastAPI backend on http://127.0.0.1:8000 ..."
"$ROOT_DIR/.venv/bin/uvicorn" backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting React frontend on http://127.0.0.1:5173 ..."
(cd "$ROOT_DIR/frontend" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "=============================================="
echo " VoiceCast is running!"
echo " Open in browser: http://127.0.0.1:5173"
echo " API docs:        http://127.0.0.1:8000/docs"
echo " Press Ctrl+C to stop both servers."
echo "=============================================="

wait
