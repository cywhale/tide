#!/bin/bash

set -e

DASK_TLS_DIR=/home/odbadmin/.config/dask/tls
export DASK_SCHEDULER_ADDRESS=tls://127.0.0.1:8786
export DASK_TLS_CA_FILE="$DASK_TLS_DIR/certs/ca.crt"
export DASK_TLS_CLIENT_CERT="$DASK_TLS_DIR/certs/tide-client.crt"
export DASK_TLS_CLIENT_KEY="$DASK_TLS_DIR/private/tide-client.key"
for tls_file in "$DASK_TLS_CA_FILE" "$DASK_TLS_CLIENT_CERT" "$DASK_TLS_CLIENT_KEY"; do
  if [ ! -r "$tls_file" ]; then
    echo "Required Dask TLS file is not readable: $tls_file" >&2
    exit 1
  fi
done

# Preserve this VM's existing production runtime: the active Tide process used
# the py311 Gunicorn environment. Do not let `uv run` create/sync a new .venv
# merely because PM2's updated PATH happens to include uv.
if [ -z "${GUNICORN_BIN:-}" ] && [ -x /home/odbadmin/.pyenv/versions/py311/bin/gunicorn ]; then
  GUNICORN_BIN=/home/odbadmin/.pyenv/versions/py311/bin/gunicorn
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
