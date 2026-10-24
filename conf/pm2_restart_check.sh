#!/bin/bash
# Restart a PM2 app with its stored environment and verify the restart took
# effect. Exits non-zero unless every check passes, so a following command
# chained with && never runs after a failed or ineffective restart.
#
#   pm2_restart_check.sh [--pm2 PATH] [--pm2-home DIR] [--name tide]
#                        [--expect-exec PATH] [--timeout 60] [--settle 10]
#
# The target daemon is always --pm2-home (default: production
# /home/odbadmin/.pm2). An inherited PM2_HOME is ignored, so a shell left
# pointing at a test daemon cannot redirect the restart. Before any pm2 call
# the daemon must be proven alive (see daemon_alive); otherwise the script
# stops without calling pm2, so a missing or stale daemon never leads pm2 to
# start a new one. Limit: if the daemon exits after that check but before a
# pm2 call reaches it (a narrow race), pm2 can still start a new daemon. The
# script cannot prevent that, but re-checks at the end and fails unless the
# same daemon PID is still alive, so such a run is never reported as OK.
#
# Checks: `pm2 restart` exits 0 (no --update-env, so the operator's shell
# variables never reach production); within --timeout seconds the app is
# online with a new PID; the restart count rose by exactly one; after --settle
# seconds the PID and restart count are unchanged (no crash loop); and the
# process command line contains --expect-exec (the py311 Gunicorn).
set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

die() { echo "FAIL: $*" >&2; exit 1; }
usage() { sed -n '2,23p' "$0" >&2; exit 2; }

PM2=$HOME/.npm-global/bin/pm2
TARGET_HOME=/home/odbadmin/.pm2
NAME=tide
EXPECT_EXEC=/home/odbadmin/.pyenv/versions/py311/bin/gunicorn
TIMEOUT=60
SETTLE=10
while [ $# -gt 0 ]; do
  case "$1" in
    --pm2) PM2=${2:?--pm2 needs a value}; shift 2 ;;
    --pm2-home) TARGET_HOME=${2:?--pm2-home needs a value}; shift 2 ;;
    --name) NAME=${2:?--name needs a value}; shift 2 ;;
    --expect-exec) EXPECT_EXEC=${2:?--expect-exec needs a value}; shift 2 ;;
    --timeout) TIMEOUT=${2:?--timeout needs a value}; shift 2 ;;
    --settle) SETTLE=${2:?--settle needs a value}; shift 2 ;;
    *) usage ;;
  esac
done

TARGET_HOME=${TARGET_HOME%/}

# Fail-closed liveness check that never runs pm2: rpc.sock is a socket, the
# PID in pm2.pid is alive, that process is the PM2 God Daemon for this exact
# PM2_HOME (PM2 titles it "PM2 vX: God Daemon (<PM2_HOME>)"), and rpc.sock
# accepts a connection. A stale socket, a plain file, a dead or reused PID,
# or another home's daemon all fail. Prints the reason on failure.
daemon_alive() {
  local home=$1 pid args
  [ -S "$home/rpc.sock" ] || { echo "rpc.sock is missing or not a socket"; return 1; }
  [ -r "$home/pm2.pid" ] || { echo "pm2.pid is missing"; return 1; }
  pid=$(tr -dc '0-9' < "$home/pm2.pid")
  [ -n "$pid" ] || { echo "pm2.pid holds no PID"; return 1; }
  kill -0 "$pid" 2>/dev/null || { echo "PID $pid from pm2.pid is not running"; return 1; }
  args=$(ps -o args= -p "$pid") || { echo "cannot read the command line of PID $pid"; return 1; }
  case "$args" in
    *"God Daemon ($home)"*) ;;
    *) echo "PID $pid is not the PM2 daemon for $home: $args"; return 1 ;;
  esac
  python3 -I -c '
import socket, sys
s = socket.socket(socket.AF_UNIX)
s.settimeout(3)
s.connect(sys.argv[1])
s.close()
' "$home/rpc.sock" 2>/dev/null || { echo "rpc.sock refuses connections"; return 1; }
}

reason=$(daemon_alive "$TARGET_HOME") \
  || die "no running PM2 daemon at $TARGET_HOME ($reason); refusing to call pm2"
DAEMON_PID=$(tr -dc '0-9' < "$TARGET_HOME/pm2.pid")
if [ -n "${PM2_HOME:-}" ] && [ "$PM2_HOME" != "$TARGET_HOME" ]; then
  echo "note: ignoring inherited PM2_HOME=$PM2_HOME" >&2
fi
export PM2_HOME=$TARGET_HOME
echo "daemon: PM2_HOME=$PM2_HOME"

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

# Detect (not prevent) the race noted in the header: the daemon must still
# be the one that passed the initial check.
reason=$(daemon_alive "$TARGET_HOME") \
  || die "PM2 daemon at $TARGET_HOME died or changed during the run ($reason); inspect it before trusting this restart"
now_pid=$(tr -dc '0-9' < "$TARGET_HOME/pm2.pid")
[ "$now_pid" = "$DAEMON_PID" ] \
  || die "PM2 daemon at $TARGET_HOME changed during the run (PID $DAEMON_PID -> $now_pid); inspect it before trusting this restart"
echo "OK restart: $NAME pid $pid0 -> $pid, restarts $restarts0 -> $restarts, stable ${SETTLE}s"
echo "command: $args"
