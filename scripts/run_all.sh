#!/bin/bash
set -e

echo "=== Assistive AI Web Application Launcher ==="

# Check Python environment
if [ ! -d "backend/venv" ]; then
    echo "[INFO] Creating Python virtual environment in backend/venv..."
    python3 -m venv backend/venv
fi

echo "[INFO] Activating virtual environment and installing backend requirements..."
source backend/venv/bin/activate
pip install -q --upgrade pip
pip install -q -r backend/requirements.txt

# Start Backend Server in Background
echo "[INFO] Launching FastAPI backend server on http://localhost:8000..."
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Function to kill backend on exit
cleanup() {
    echo "\n[INFO] Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
}
trap cleanup EXIT

# Start Frontend
echo "[INFO] Starting Next.js frontend server on http://localhost:3000..."
npm --prefix frontend run dev
