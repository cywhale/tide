#!/bin/bash
# Restart a PM2 app with its stored environment and verify the restart took
# effect. Exits non-zero unless every check passes, so a following command
# chained with && never runs after a failed or ineffective restart.
#
#   pm2_restart_check.sh [--pm2 PATH] [--name tide] [--expect-exec PATH]
#                        [--timeout 60] [--settle 10]
#
# Checks: `pm2 restart` exits 0 (no --update-env, so the operator's shell
# variables never reach production); within --timeout seconds the app is
# online with a new PID; the restart count rose by exactly one; after --settle
# seconds the PID and restart count are unchanged (no crash loop); and the
# process command line contains --expect-exec (the py311 Gunicorn).
set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

die() { echo "FAIL: $*" >&2; exit 1; }
usage() { sed -n '2,13p' "$0" >&2; exit 2; }

PM2=$HOME/.npm-global/bin/pm2
NAME=tide
EXPECT_EXEC=/home/odbadmin/.pyenv/versions/py311/bin/gunicorn
TIMEOUT=60
SETTLE=10
while [ $# -gt 0 ]; do
  case "$1" in
    --pm2) PM2=${2:?--pm2 needs a value}; shift 2 ;;
    --name) NAME=${2:?--name needs a value}; shift 2 ;;
    --expect-exec) EXPECT_EXEC=${2:?--expect-exec needs a value}; shift 2 ;;
    --timeout) TIMEOUT=${2:?--timeout needs a value}; shift 2 ;;
    --settle) SETTLE=${2:?--settle needs a value}; shift 2 ;;
    *) usage ;;
  esac
done

# Prints "STATUS PID RESTARTS" for exactly one app named NAME.
state() {
  "$PM2" jlist | python3 -I -c '
import json, sys
apps = [p for p in json.load(sys.stdin) if p.get("name") == sys.argv[1]]
if len(apps) != 1:
    sys.exit("expected exactly one PM2 app named %r, found %d" % (sys.argv[1], len(apps)))
e = apps[0]["pm2_env"]
print(e["status"], apps[0]["pid"], e["restart_time"])
' "$NAME"
}

before=$(state) || die "cannot read PM2 state for $NAME"
read -r status0 pid0 restarts0 <<< "$before"
echo "before: status=$status0 pid=$pid0 restarts=$restarts0"

"$PM2" restart "$NAME" || die "pm2 restart $NAME exited non-zero"

waited=0
while :; do
  cur=$(state) || die "cannot read PM2 state for $NAME after restart"
  read -r status pid restarts <<< "$cur"
  if [ "$status" = "online" ] && [ "$pid" != "$pid0" ] && [ "$pid" != "0" ]; then
    break
  fi
  [ "$waited" -lt "$TIMEOUT" ] \
    || die "not online with a new PID after ${TIMEOUT}s (status=$status pid=$pid, before pid=$pid0)"
  sleep 1
  waited=$((waited + 1))
done
[ "$restarts" -eq $((restarts0 + 1)) ] \
  || die "restart count went from $restarts0 to $restarts, expected exactly +1"

sleep "$SETTLE"
cur=$(state) || die "cannot read PM2 state for $NAME after settling"
read -r status2 pid2 restarts2 <<< "$cur"
[ "$status2 $pid2 $restarts2" = "$status $pid $restarts" ] \
  || die "not stable after ${SETTLE}s: status=$status2 pid=$pid2 restarts=$restarts2 (was $status $pid $restarts)"

args=$(ps -o args= -p "$pid") || die "cannot read the command line of PID $pid"
case " $args " in
  *" $EXPECT_EXEC "*) ;;
  *) die "PID $pid runs '$args', expected $EXPECT_EXEC" ;;
esac
echo "OK restart: $NAME pid $pid0 -> $pid, restarts $restarts0 -> $restarts, stable ${SETTLE}s"
echo "command: $args"
