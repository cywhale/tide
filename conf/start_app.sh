#!/bin/bash

set -e

# Tide no longer uses the shared Dask service. Do not pass inherited client
# credentials or scheduler settings to the API process.
unset DASK_SCHEDULER_ADDRESS DASK_TLS_CA_FILE DASK_TLS_CLIENT_CERT DASK_TLS_CLIENT_KEY

# Production runs only the VM24 py311 Gunicorn, or an explicitly configured
# absolute GUNICORN_BIN. There is deliberately no `uv run` or PATH fallback:
# a PM2 start must never create/sync a uv environment or pick up another
# interpreter by accident.
PY311_GUNICORN=/home/odbadmin/.pyenv/versions/py311/bin/gunicorn
if [ -n "${GUNICORN_BIN:-}" ]; then
  case "$GUNICORN_BIN" in
    /*) ;;
    *)
      echo "GUNICORN_BIN must be an absolute path: $GUNICORN_BIN" >&2
      exit 127
      ;;
  esac
elif [ -x "$PY311_GUNICORN" ]; then
  GUNICORN_BIN=$PY311_GUNICORN
else
  echo "VM24 py311 Gunicorn not found: $PY311_GUNICORN (set an absolute GUNICORN_BIN)" >&2
  exit 127
fi
if [ ! -x "$GUNICORN_BIN" ]; then
  echo "GUNICORN_BIN is not executable: $GUNICORN_BIN" >&2
  exit 127
fi

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

exec "$GUNICORN_BIN" "${GUNICORN_ARGS[@]}"
