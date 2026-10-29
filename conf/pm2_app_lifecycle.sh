#!/bin/bash
# Stop or start one PM2 app and verify the result, for the stop-first py311
# Zarr maintenance. Exits non-zero unless every check passes, so a command
# chained after it with && never runs after a failed or partial step.
#
#   pm2_app_lifecycle.sh stop  --name APP --match WORD [--expect-exec WORD]
#   pm2_app_lifecycle.sh start --name APP --match WORD [--expect-exec WORD]
#   pm2_app_lifecycle.sh pid    --name APP
#   pm2_app_lifecycle.sh status --name APP
#       common: [--pm2 PATH] [--pm2-home DIR] [--timeout 60] [--settle 10]
#
# WORD arguments are compared with whole space-separated words of a command
# line. --match names a word that every process of the app carries and no
# other process does (for example tide_app:app, mhw_app:app, or the py311
# dask-worker path). A "match root" is a matching process whose parent does
# not match; Gunicorn workers repeat their master's command line, so one
# running app has exactly one root. --expect-exec must be a word of the
# command line of the PID that PM2 manages (the Gunicorn master for tide, the
# bash launcher for mhwapi).
#
# stop:  exactly one app is named APP and it is not stopped. When it is
#   online, its PID runs --expect-exec and its process tree holds the only
#   match root; an app PM2 marked errored after a crash loop (PID 0) is
#   stopped too, so that recovery can bring it to a known state. `pm2 stop
#   APP` exits 0. Within --timeout the
#   app is stopped with PID 0 and every process of the tree recorded before
#   the stop has exited. After --settle seconds the app is still stopped and
#   no process matches (no respawn by PM2 or by a nanny, no orphan).
# start: exactly one app is named APP, it is stopped (or errored), and no
#   process matches.
#   It runs `pm2 restart APP`: on a stopped app PM2 starts it with its stored
#   environment (no --update-env), and unlike `pm2 start APP` it never
#   creates a new app when the name is unknown. Within --timeout the app is
#   online with a non-zero PID whose tree holds the only match root. After
#   --settle seconds the status, PID, and restart count are unchanged (no
#   crash loop). The restart count is not expected to rise: PM2 5.4.2 counts
#   restarts only of apps that were online.
# pid:    prints the PID of APP when it is online; never changes anything.
# status: prints "STATUS PID RESTARTS" of APP; never changes anything.
# Both stop and start also require every other PM2 app to keep its status
# and PID, and the daemon to be the same one at the end.
#
# The target daemon is always --pm2-home (default: production
# /home/odbadmin/.pm2); an inherited PM2_HOME is ignored. pm2 is never called
# unless the daemon is proven alive first (same check as
# pm2_restart_check.sh), so a missing daemon is never replaced by a new one.
set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

die() { echo "FAIL: $*" >&2; exit 1; }
usage() { sed -n '2,46p' "$0" >&2; exit 2; }

[ $# -ge 1 ] || usage
MODE=$1
shift
PM2=$HOME/.npm-global/bin/pm2
TARGET_HOME=/home/odbadmin/.pm2
NAME=""
MATCH=""
EXPECT_EXEC=""
TIMEOUT=60
SETTLE=10
while [ $# -gt 0 ]; do
  case "$1" in
    --pm2) PM2=${2:?--pm2 needs a value}; shift 2 ;;
    --pm2-home) TARGET_HOME=${2:?--pm2-home needs a value}; shift 2 ;;
    --name) NAME=${2:?--name needs a value}; shift 2 ;;
    --match) MATCH=${2:?--match needs a value}; shift 2 ;;
    --expect-exec) EXPECT_EXEC=${2:?--expect-exec needs a value}; shift 2 ;;
    --timeout) TIMEOUT=${2:?--timeout needs a value}; shift 2 ;;
    --settle) SETTLE=${2:?--settle needs a value}; shift 2 ;;
    *) usage ;;
  esac
done
case "$MODE" in
  stop|start) [ -n "$NAME" ] && [ -n "$MATCH" ] || usage ;;
  pid|status) [ -n "$NAME" ] || usage ;;
  *) usage ;;
esac
TARGET_HOME=${TARGET_HOME%/}

# Fail-closed liveness check that never runs pm2; identical to the one in
# pm2_restart_check.sh. Prints the verified daemon PID.
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
  echo "$pid"
}

out=$(daemon_alive "$TARGET_HOME") \
  || die "no running PM2 daemon at $TARGET_HOME ($out); refusing to call pm2"
DAEMON_PID=$out
if [ -n "${PM2_HOME:-}" ] && [ "$PM2_HOME" != "$TARGET_HOME" ]; then
  echo "note: ignoring inherited PM2_HOME=$PM2_HOME" >&2
fi
export PM2_HOME=$TARGET_HOME

# apps: one "NAME STATUS PID RESTARTS" line per PM2 app, sorted.
apps() {
  "$PM2" jlist | python3 -I -c '
import json, sys
for p in json.load(sys.stdin):
    e = p["pm2_env"]
    print(p["name"], e["status"], p["pid"], e["restart_time"])
' | LC_ALL=C sort
}

# state: "STATUS PID RESTARTS" of the one app named NAME.
state() {
  local all n
  all=$(apps) || return 1
  n=$(awk -v n="$NAME" '$1 == n' <<< "$all" | wc -l | tr -d ' ')
  [ "$n" = 1 ] || { echo "expected exactly one PM2 app named $NAME, found $n" >&2; return 1; }
  awk -v n="$NAME" '$1 == n {print $2, $3, $4}' <<< "$all"
}

others() { apps | awk -v n="$NAME" '$1 != n {print $1, $2, $3}'; }

if [ "$MODE" = status ]; then
  state || die "cannot read PM2 state for $NAME"
  exit 0
fi
if [ "$MODE" = pid ]; then
  cur=$(state) || die "cannot read PM2 state for $NAME"
  read -r status pid _ <<< "$cur"
  [ "$status" = online ] && [ "$pid" != 0 ] || die "$NAME is $status (pid $pid), not online"
  echo "$pid"
  exit 0
fi

# Process table as "PID PPID ARGS..." lines.
proc_table() { ps -eo pid=,ppid=,args=; }

# tree ROOT: ROOT and all its descendants, one PID per line.
tree() {
  proc_table | python3 -I -c '
import sys
root = sys.argv[1]
kids = {}
for line in sys.stdin:
    f = line.split(None, 2)
    if len(f) >= 2:
        kids.setdefault(f[1], []).append(f[0])
seen, todo = [], [root]
while todo:
    p = todo.pop()
    if p not in seen:
        seen.append(p)
        todo.extend(kids.get(p, []))
print("\n".join(seen))
' "$1"
}

# match_roots: PIDs of matching processes whose parent does not match. This
# script, its ancestors (an operator's `bash -c` line carries the same
# words), and its descendants (the ps | python3 pipeline) never count.
match_roots() {
  proc_table | python3 -I -c '
import sys
word, me = sys.argv[1], sys.argv[2]
rows = {}
for line in sys.stdin:
    f = line.split(None, 2)
    if len(f) == 3:
        rows[f[0]] = (f[1], word in f[2].split())
def chain(pid):
    seen = []
    while pid in rows and pid not in seen:
        seen.append(pid)
        pid = rows[pid][0]
    return seen
mine = set(chain(me)) | {p for p in rows if me in chain(p)}
for pid, (ppid, hit) in rows.items():
    if hit and pid not in mine and not (rows.get(ppid, ("", False))[1] and ppid not in mine):
        print(pid)
' "$MATCH" "$$"
}

# check_roots PID: the only match root lies in the tree of PID.
check_roots() {
  local roots t
  roots=$(match_roots)
  [ "$(grep -c . <<< "$roots" || true)" = 1 ] \
    || die "expected one process tree matching '$MATCH', found roots: $(echo $roots)"
  t=$(tree "$1")
  grep -qx "$roots" <<< "$t" \
    || die "the process matching '$MATCH' (PID $roots) is not in the tree of $NAME (PID $1)"
}

check_exec() {
  local args
  [ -n "$EXPECT_EXEC" ] || return 0
  args=$(ps -o args= -p "$1") || die "cannot read the command line of PID $1"
  case " $args " in
    *" $EXPECT_EXEC "*) ;;
    *) die "PID $1 runs '$args', expected $EXPECT_EXEC" ;;
  esac
}

check_daemon_same() {
  local now
  now=$(daemon_alive "$TARGET_HOME") \
    || die "PM2 daemon at $TARGET_HOME died or changed during the run ($now)"
  [ "$now" = "$DAEMON_PID" ] \
    || die "PM2 daemon at $TARGET_HOME changed during the run (PID $DAEMON_PID -> $now)"
}

others0=$(others) || die "cannot read the PM2 app list"
cur=$(state) || die "cannot read PM2 state for $NAME"
read -r status0 pid0 restarts0 <<< "$cur"
echo "before: $NAME status=$status0 pid=$pid0 restarts=$restarts0"

if [ "$MODE" = stop ]; then
  [ "$status0" != stopped ] || die "$NAME is already stopped; nothing to stop"
  tree0=""
  if [ "$status0" = online ] && [ "$pid0" != 0 ]; then
    check_exec "$pid0"
    check_roots "$pid0"
    tree0=$(tree "$pid0")
  elif [ "$pid0" != 0 ]; then
    die "$NAME is $status0 with pid $pid0; stop it only when online or errored"
  fi
  "$PM2" stop "$NAME" || die "pm2 stop $NAME exited non-zero"
  waited=0
  while :; do
    cur=$(state) || die "cannot read PM2 state for $NAME after stop"
    read -r status pid restarts <<< "$cur"
    alive=""
    for p in $tree0; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
    [ "$status" = stopped ] && [ "$pid" = 0 ] && [ -z "$alive" ] && break
    [ "$waited" -lt "$TIMEOUT" ] \
      || die "not fully stopped after ${TIMEOUT}s (status=$status pid=$pid; still running:${alive:- none})"
    sleep 1
    waited=$((waited + 1))
  done
  sleep "$SETTLE"
  cur=$(state) || die "cannot read PM2 state for $NAME after settling"
  [ "$cur" = "stopped 0 $restarts" ] \
    || die "not stable after ${SETTLE}s: $cur (was stopped 0 $restarts); PM2 restarted it"
  roots=$(match_roots)
  [ -z "$roots" ] \
    || die "processes matching '$MATCH' still run after the stop: $(echo $roots)"
  result="OK stop: $NAME pid $pid0 stopped, ${SETTLE}s without respawn, no process matches $MATCH"
else
  case "$status0" in
    stopped|errored) [ "$pid0" = 0 ] || die "$NAME is $status0 with pid $pid0" ;;
    *) die "$NAME is $status0 (pid $pid0), not stopped; this script starts only stopped apps" ;;
  esac
  roots=$(match_roots)
  [ -z "$roots" ] \
    || die "processes matching '$MATCH' already run before the start: $(echo $roots)"
  "$PM2" restart "$NAME" || die "pm2 restart $NAME exited non-zero"
  waited=0
  while :; do
    cur=$(state) || die "cannot read PM2 state for $NAME after start"
    read -r status pid restarts <<< "$cur"
    [ "$status" = online ] && [ "$pid" != 0 ] && [ -n "$(match_roots)" ] && break
    [ "$waited" -lt "$TIMEOUT" ] \
      || die "not online with a matching process after ${TIMEOUT}s (status=$status pid=$pid)"
    sleep 1
    waited=$((waited + 1))
  done
  sleep "$SETTLE"
  cur=$(state) || die "cannot read PM2 state for $NAME after settling"
  [ "$cur" = "online $pid $restarts" ] \
    || die "not stable after ${SETTLE}s: $cur (was online $pid $restarts); crash loop or restart"
  [ "$restarts" = "$restarts0" ] \
    || die "restart count went from $restarts0 to $restarts during the start"
  check_exec "$pid"
  check_roots "$pid"
  result="OK start: $NAME online as pid $pid, stable ${SETTLE}s, restarts $restarts"
fi

others1=$(others) || die "cannot read the PM2 app list at the end"
[ "$others0" = "$others1" ] \
  || die "another PM2 app changed during the run: $(diff <(echo "$others0") <(echo "$others1") | grep '^[<>]' | tr '\n' ' ')"
check_daemon_same
echo "$result"
