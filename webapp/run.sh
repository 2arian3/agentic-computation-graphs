#!/usr/bin/env bash
# One-command launcher for the ACG dashboard (production mode: FastAPI serves the SPA).
#
#   webapp/run.sh                 # build frontend if needed, then serve on :8100
#   PORT=9000 webapp/run.sh       # different port
#   SKIP_BUILD=1 webapp/run.sh    # don't (re)build the frontend
#
# The model server (vLLM) is expected on :8000; this app uses :8100 to avoid the clash.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-./.venv/bin/python}"
PORT="${PORT:-8100}"
FRONTEND="webapp/frontend"

# 1. backend deps (idempotent)
if ! "$PY" -c "import fastapi, uvicorn, sse_starlette" >/dev/null 2>&1; then
  echo "[run] installing backend deps into the venv…"
  "${PY%python}pip" install -q -r webapp/requirements.txt
fi

# 2. build the frontend if a build is missing (needs Node >= 18)
if [[ "${SKIP_BUILD:-0}" != "1" && ! -f "$FRONTEND/dist/index.html" ]]; then
  if command -v npm >/dev/null 2>&1; then
    echo "[run] building frontend…"
    ( cd "$FRONTEND" && npm install --no-audit --no-fund && npm run build )
  else
    echo "[run] WARNING: no Node/npm and no prebuilt frontend at $FRONTEND/dist."
    echo "       Install Node >= 18 and run 'cd $FRONTEND && npm install && npm run build',"
    echo "       or use the dev server (see webapp/README.md). Serving API only."
  fi
fi

# 3. serve
echo "[run] serving on http://127.0.0.1:$PORT  (Ctrl-C to stop)"
exec "$PY" -m uvicorn webapp.backend.main:app --host 0.0.0.0 --port "$PORT"
