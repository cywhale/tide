#!/bin/bash

set -e

GUNICORN_ARGS=(
  tide_app:app
  -w 2
  -k uvicorn.workers.UvicornWorker
  -b 127.0.0.1:8040
  --keyfile conf/privkey.pem
  --certfile conf/fullchain.pem
  --timeout 180
  --reload
)

if [ -n "${GUNICORN_BIN:-}" ]; then
  exec "$GUNICORN_BIN" "${GUNICORN_ARGS[@]}"
elif command -v uv >/dev/null 2>&1 && [ -f uv.lock ]; then
  exec uv run gunicorn "${GUNICORN_ARGS[@]}"
elif command -v gunicorn >/dev/null 2>&1; then
  exec gunicorn "${GUNICORN_ARGS[@]}"
elif [ -x /home/odbadmin/.pyenv/versions/py311/bin/gunicorn ]; then
  exec /home/odbadmin/.pyenv/versions/py311/bin/gunicorn "${GUNICORN_ARGS[@]}"
else
  echo "gunicorn not found. Set GUNICORN_BIN or install runtime dependencies." >&2
  exit 127
fi
