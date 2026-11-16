#!/usr/bin/env bash
# Launch the Track A streaming-ATD demo locally.
# Requires fastapi + uvicorn in the dysarthria conda env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${PROJECT_ROOT}"
exec /c/Users/Dell/.conda/envs/dysarthria/python.exe -m uvicorn src.streaming.server:app --host 0.0.0.0 --port 8765
