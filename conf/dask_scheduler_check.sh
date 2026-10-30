#!/bin/bash
# Read-only checks of the local Dask scheduler for the stop-first py311 Zarr
# maintenance. It reads only the scheduler's own HTTP monitoring routes (GET
# /health, /metrics, /json/identity.json on the loopback dashboard), so it
# never registers a Dask client, submits a task, or changes scheduler state.
# Exits non-zero unless the condition holds within --timeout seconds.
#
#   dask_scheduler_check.sh drained [--url URL] [--timeout 120]
#   dask_scheduler_check.sh ready --workers N [--nanny-pid PID] [--clients N]
#                                 [--url URL] [--timeout 60]
#
# drained: no client is connected and the scheduler knows no task in any
#   state. Use it after both APIs are stopped and before the worker stops.
# ready:   the scheduler is healthy with exactly N workers. With --nanny-pid,
#   every worker reports a nanny whose port PID listens on (ss, or lsof where
#   ss is missing); with --clients, exactly that many clients are connected.
#
# Client and task counts come from /metrics, which leaves out the
# scheduler's built-in "fire-and-forget" client. URL defaults to
# http://127.0.0.1:8787 and must be plain http on 127.0.0.1.
set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

die() { echo "FAIL: $*" >&2; exit 1; }
usage() { sed -n '2,20p' "$0" >&2; exit 2; }

[ $# -ge 1 ] || usage
MODE=$1
shift
URL=http://127.0.0.1:8787
TIMEOUT=""
WORKERS=""
NANNY_PID=""
CLIENTS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL=${2:?--url needs a value}; shift 2 ;;
    --timeout) TIMEOUT=${2:?--timeout needs a value}; shift 2 ;;
    --workers) WORKERS=${2:?--workers needs a value}; shift 2 ;;
    --nanny-pid) NANNY_PID=${2:?--nanny-pid needs a value}; shift 2 ;;
    --clients) CLIENTS=${2:?--clients needs a value}; shift 2 ;;
    *) usage ;;
  esac
done
case "$MODE" in
  drained) TIMEOUT=${TIMEOUT:-120} ;;
  ready) [ -n "$WORKERS" ] || usage; TIMEOUT=${TIMEOUT:-60} ;;
  *) usage ;;
esac
case "$URL" in
  http://127.0.0.1:[0-9]*) ;;
  *) die "--url must be http://127.0.0.1:PORT, got $URL" ;;
esac
URL=${URL%/}

get() { curl -sS --proto '=http' --max-time 5 "$URL$1"; }

# port_owner PORT: PIDs listening on 127.0.0.1:PORT.
port_owner() {
  if command -v ss >/dev/null; then
    ss -ltnpH "sport = :$1" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
  elif command -v lsof >/dev/null; then
    lsof -nP -iTCP:"$1" -sTCP:LISTEN -t | sort -u
  else
    echo "neither ss nor lsof is available" >&2
    return 1
  fi
}

# probe: one observation. Prints a summary and returns 0 when it holds.
probe() {
  local health metrics ident
  health=$(get /health) || { echo "scheduler HTTP not reachable at $URL"; return 1; }
  [ "$health" = ok ] || { echo "health is '$health'"; return 1; }
  metrics=$(get /metrics) || { echo "cannot read /metrics"; return 1; }
  ident=$(get /json/identity.json) || { echo "cannot read /json/identity.json"; return 1; }
  python3 -I - "$MODE" "$WORKERS" "$CLIENTS" "$metrics" "$ident" <<'PY'
import json, sys
mode, want_workers, want_clients, metrics, ident = sys.argv[1:]
vals = {}
for line in metrics.splitlines():
    if line.startswith("dask_scheduler_clients ") or line.startswith("dask_scheduler_tasks{") \
            or line.startswith("dask_scheduler_workers{"):
        key, _, num = line.rpartition(" ")
        vals[key] = float(num)
if "dask_scheduler_clients" not in vals or not any(k.startswith("dask_scheduler_tasks{") for k in vals):
    print("metrics lack dask_scheduler_clients or dask_scheduler_tasks")
    sys.exit(1)
clients = int(vals["dask_scheduler_clients"])
tasks = int(sum(v for k, v in vals.items() if k.startswith("dask_scheduler_tasks{")))
workers = json.loads(ident)["workers"]
summary = f"clients={clients} tasks={tasks} workers={len(workers)}"
if mode == "drained":
    ok = clients == 0 and tasks == 0
else:
    ok = len(workers) == int(want_workers) and (not want_clients or clients == int(want_clients))
    for addr, w in sorted(workers.items()):
        summary += f" worker={addr} status={w.get('status')} nanny={w.get('nanny')}"
        ok = ok and w.get("status") == "running"
print(summary)
sys.exit(0 if ok else 1)
PY
}

# nanny_ok: every worker's nanny port is owned by NANNY_PID.
nanny_ok() {
  local ident ports port owners
  ident=$(get /json/identity.json) || { echo "cannot read /json/identity.json"; return 1; }
  ports=$(python3 -I -c '
import json, sys
for w in json.loads(sys.argv[1])["workers"].values():
    n = w.get("nanny") or ""
    print(n.rsplit(":", 1)[-1] if ":" in n else "none")
' "$ident") || return 1
  for port in $ports; do
    [ "$port" != none ] || { echo "a worker reports no nanny"; return 1; }
    owners=$(port_owner "$port") || return 1
    owners=$(echo $owners)
    [ "$owners" = "$NANNY_PID" ] \
      || { echo "nanny port $port is owned by [$owners], expected $NANNY_PID"; return 1; }
  done
}

waited=0
while :; do
  if summary=$(probe); then
    if [ "$MODE" = ready ] && [ -n "$NANNY_PID" ]; then
      if msg=$(nanny_ok); then break; fi
      summary="$summary; $msg"
    else
      break
    fi
  fi
  [ "$waited" -lt "$TIMEOUT" ] || die "$MODE not reached after ${TIMEOUT}s: $summary"
  sleep 1
  waited=$((waited + 1))
done
echo "OK $MODE: $summary${NANNY_PID:+; nanny PID $NANNY_PID owns the nanny port of every worker}"
