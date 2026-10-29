#!/bin/bash
# Isolated rehearsal of the stop-first Zarr maintenance sequence with real
# PM2, a real Dask scheduler and nanny-managed worker, and two dummy Gunicorn
# apps shaped like Tide and MHW. Never run it on VM24: it refuses when the
# production PM2 home exists, and every pm2 call uses its own fresh PM2_HOME.
#
#   stop_first_rehearsal.sh --pm2 PATH --venv DIR [--keep]
#
# PATH is a PM2 5.4.2 binary; DIR is a Python 3.11 venv with dask and
# distributed 2025.3.0, gunicorn 23.0.0 and prometheus_client (the py311
# versions on VM24 as of 2026-10-28). Nothing outside a fresh /tmp/odbz.*
# directory is created; the apps listen on 127.0.0.1:18030/18040/18786/18787.
#
# Differences from VM24: plain tcp:// instead of mutual TLS, a sync WSGI
# worker instead of uvicorn, no nginx, and no real Zarr or MHW code. The
# rehearsal tests process lifecycle, not the applications.
set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

die() { echo "REHEARSAL FAIL: $*" >&2; exit 1; }
PM2_BIN=""
VENV=""
KEEP=no
while [ $# -gt 0 ]; do
  case "$1" in
    --pm2) PM2_BIN=${2:?}; shift 2 ;;
    --venv) VENV=${2:?}; shift 2 ;;
    --keep) KEEP=yes; shift ;;
    *) sed -n '2,16p' "$0" >&2; exit 2 ;;
  esac
done
[ -x "$PM2_BIN" ] && [ -x "$VENV/bin/dask-scheduler" ] || die "need --pm2 and --venv"
[ ! -e /home/odbadmin/.pm2 ] || die "/home/odbadmin/.pm2 exists: this looks like VM24; refusing"
HERE=$(cd "$(dirname "$0")" && pwd)
CONF=$(cd "$HERE/../../conf" && pwd)
LIFE=$CONF/pm2_app_lifecycle.sh
DASKC=$CONF/dask_scheduler_check.sh

W=$(mktemp -d /tmp/odbz.XXXXXX)
export PM2_HOME=$W/pm2
mkdir -p "$PM2_HOME" "$W/app" "$W/log"
pm2() { PM2_HOME=$W/pm2 "$PM2_BIN" "$@"; }
cleanup() {
  pm2 kill >/dev/null 2>&1 || true
  [ -z "${BYSTANDER:-}" ] || kill "$BYSTANDER" 2>/dev/null || true
  [ -z "${LOAD:-}" ] || kill "$LOAD" 2>/dev/null || true
  [ -z "${BLOCKER:-}" ] || kill "$BLOCKER" 2>/dev/null || true
  [ -z "${EXTRA:-}" ] || kill "$EXTRA" 2>/dev/null || true
  if [ "$KEEP" = yes ]; then echo "kept $W"; fi
}
trap cleanup EXIT

SCHED=tcp://127.0.0.1:18786
cat > "$W/app/tide_app.py" <<EOF
from distributed import Client
try:    # like Tide: a failed connection is logged and the app still serves
    client = Client("$SCHED", name="tideapi", set_as_default=True, timeout="5s")
except Exception as exc:
    print("Dask client connection failed:", exc, flush=True)
def app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"tide ok\n"]
EOF
cat > "$W/app/mhw_app.py" <<EOF
import dask.array as da
from distributed import Client
try:    # like MHW: without a scheduler it computes in-process
    client = Client("$SCHED", name="mhwapi", set_as_default=True, timeout="5s")
except Exception as exc:
    print("Dask scheduler unavailable; falling back to local execution.", flush=True)
def app(environ, start_response):
    # Like MHW: every data request computes through the default client.
    x = da.ones((2000, 2000), chunks=(250, 250)).sum().compute()
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"mhw %d\n" % int(x)]
EOF
# Tide's launcher execs Gunicorn (PM2 manages the master); MHW's runs it as
# a child of bash (PM2 manages the launcher), as on VM24.
cat > "$W/app/tide_start.sh" <<EOF
#!/bin/bash
cd "$W/app"
exec "$VENV/bin/gunicorn" tide_app:app -w 2 -b 127.0.0.1:18040 --reload
EOF
cat > "$W/app/mhw_start.sh" <<EOF
#!/bin/bash
cd "$W/app"
"$VENV/bin/gunicorn" mhw_app:app -w 2 -b 127.0.0.1:18030 --reload
EOF
chmod +x "$W/app/tide_start.sh" "$W/app/mhw_start.sh"
cat > "$W/ecosystem.config.cjs" <<EOF
module.exports = { apps: [
  { name: 'dask-scheduler', script: '/bin/bash', interpreter: 'none', autorestart: true,
    args: ['-c', 'exec $VENV/bin/dask-scheduler --host 127.0.0.1 --port 18786 --dashboard-address 127.0.0.1:18787'],
    out_file: '$W/log/sched.out', error_file: '$W/log/sched.err' },
  { name: 'dask-worker', script: '/bin/bash', interpreter: 'none', autorestart: true,
    args: ['-c', 'exec $VENV/bin/dask-worker $SCHED --host 127.0.0.1 --memory-limit 8GB --no-dashboard'],
    out_file: '$W/log/worker.out', error_file: '$W/log/worker.err' },
  { name: 'tide', script: '$W/app/tide_start.sh', interpreter: 'bash', autorestart: true, max_memory_restart: '4G',
    pre_stop: "touch $W/PRE_STOP_RAN",
    out_file: '$W/log/tide.out', error_file: '$W/log/tide.err' },
  { name: 'mhwapi', script: '$W/app/mhw_start.sh', interpreter: 'bash', autorestart: true,
    out_file: '$W/log/mhw.out', error_file: '$W/log/mhw.err' },
]};
EOF

step() { echo; echo "== $*"; }
L() { "$LIFE" "$@" --pm2 "$PM2_BIN" --pm2-home "$PM2_HOME" --timeout 60 --settle 8; }
D() { "$DASKC" "$@" --url http://127.0.0.1:18787; }
http_ok() { curl -sS --proto '=http' --max-time 30 "http://127.0.0.1:$1/" | grep -q "$2"; }

step "start the isolated daemon: Dask first, then the two APIs"
pm2 start "$W/ecosystem.config.cjs" --only dask-scheduler,dask-worker >/dev/null
D ready --workers 1 --timeout 60
pm2 start "$W/ecosystem.config.cjs" --only tide,mhwapi >/dev/null
for i in $(seq 60); do http_ok 18030 "mhw 4000000" && http_ok 18040 "tide ok" && break; sleep 1; done
http_ok 18030 "mhw 4000000" && http_ok 18040 "tide ok" || die "apps did not come up"
D ready --workers 1 --clients 4 --timeout 60

# A process outside PM2 whose command line is exactly "tide_app 600": the
# pre_stop pattern (grep -w tide_app) would kill it if PM2 ran pre_stop.
bash -c 'exec -a tide_app sleep 600' &
BYSTANDER=$!
# Continuous MHW traffic, so the scheduler has tasks while the APIs run.
( while :; do curl -sS --max-time 30 http://127.0.0.1:18030/ >/dev/null 2>&1 || true; done ) &
LOAD=$!
sleep 3

step "stop the APIs (MHW first: it submits Dask work), then drain"
L stop --name mhwapi --match mhw_app:app --expect-exec "$W/app/mhw_start.sh"
L stop --name tide --match tide_app:app --expect-exec "$VENV/bin/gunicorn"
kill "$LOAD"; LOAD=""
D drained --timeout 60

step "stop the Dask worker (nanny) and the scheduler"
L stop --name dask-worker --match "$VENV/bin/dask-worker"
L stop --name dask-scheduler --match "$VENV/bin/dask-scheduler"
step "nothing runs from the venv (the quiescence check before apply)"
if pgrep -f -- "$VENV/" >/dev/null; then
  ps -eo pid=,ppid=,args= | grep -F -- "$VENV/" | grep -v grep
  die "processes still run from the venv"
fi
echo "OK quiescent: no process runs from $VENV"
[ ! -e "$W/PRE_STOP_RAN" ] || die "PM2 ran tide's pre_stop"
kill -0 "$BYSTANDER" || die "the tide_app bystander was killed"
echo "OK pre_stop never ran; the 'tide_app' bystander outside PM2 is alive"
grep -h "Restarting worker" "$W"/log/worker.* && echo "note: the nanny logged a worker restart during the stop" \
  || echo "OK the nanny logged no worker restart"

step "start again: scheduler, worker, Tide, MHW"
L start --name dask-scheduler --match "$VENV/bin/dask-scheduler"
D ready --workers 0 --timeout 30
L start --name dask-worker --match "$VENV/bin/dask-worker"
NANNY=$("$LIFE" pid --name dask-worker --pm2 "$PM2_BIN" --pm2-home "$PM2_HOME")
D ready --workers 1 --nanny-pid "$NANNY" --timeout 60
L start --name tide --match tide_app:app --expect-exec "$VENV/bin/gunicorn"
L start --name mhwapi --match mhw_app:app --expect-exec "$W/app/mhw_start.sh"
http_ok 18040 "tide ok" || die "tide does not answer"
http_ok 18030 "mhw 4000000" || die "mhw does not answer"
D ready --workers 1 --nanny-pid "$NANNY" --clients 4 --timeout 60

step "negative: a scheduler that cannot bind crash-loops and the start check fails"
L stop --name mhwapi --match mhw_app:app --expect-exec "$W/app/mhw_start.sh"
L stop --name tide --match tide_app:app --expect-exec "$VENV/bin/gunicorn"
D drained --timeout 60
L stop --name dask-worker --match "$VENV/bin/dask-worker"
L stop --name dask-scheduler --match "$VENV/bin/dask-scheduler"
"$VENV/bin/python" -I -c '
import socket, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 18786)); s.listen(1); time.sleep(600)' &
BLOCKER=$!
sleep 1
if L start --name dask-scheduler --match "$VENV/bin/dask-scheduler"; then
  die "the start check passed for a scheduler that cannot bind"
fi
echo "OK the start check failed for the crash-looping scheduler"
for i in $(seq 60); do
  [ "$("$LIFE" status --name dask-scheduler --pm2 "$PM2_BIN" --pm2-home "$PM2_HOME" | cut -d' ' -f1)" = errored ] && break
  sleep 1
done
L stop --name dask-scheduler --match "$VENV/bin/dask-scheduler"
echo "OK the errored scheduler was brought to stopped"
kill "$BLOCKER"; BLOCKER=""

step "negative: an extra Dask client makes the drain check fail"
L start --name dask-scheduler --match "$VENV/bin/dask-scheduler"
"$VENV/bin/python" -I -c '
import time
from distributed import Client
c = Client("'"$SCHED"'", name="unknown", timeout="5s"); time.sleep(600)' &
EXTRA=$!
for i in $(seq 30); do
  curl -sS --max-time 5 http://127.0.0.1:18787/metrics 2>/dev/null | grep -qx "dask_scheduler_clients 1.0" && break
  sleep 1
done
if D drained --timeout 5; then
  kill "$EXTRA"
  die "drained passed with an extra client connected"
fi
kill "$EXTRA"
echo "OK drained failed while an unknown client was connected"
D drained --timeout 30
L stop --name dask-scheduler --match "$VENV/bin/dask-scheduler"

step "negative: stopping an app that is already stopped is refused without calling pm2 stop"
if L stop --name dask-worker --match "$VENV/bin/dask-worker"; then
  die "stop passed for an app that was already stopped"
fi
echo "OK stop refused for a stopped app"

echo
echo "REHEARSAL PASSED"
