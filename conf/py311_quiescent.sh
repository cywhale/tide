#!/bin/bash
# Read-only: check that no live process runs or maps a file from the shared
# py311, the precondition for replacing its Zarr. Fails closed: a live
# process whose command line or memory maps cannot be read counts as
# possibly using py311.
#
#   py311_quiescent.sh [--inspectable-only] [--proc DIR] [--prefix DIR]
#
# Passes (exit 0, "OK quiescent") only when every process listed under
# /proc was either inspected or confirmed to have exited meanwhile (its
# /proc entry is gone), and none has PREFIX in its command line or in any
# mapped file. Otherwise it lists the processes that use py311 and those it
# could not inspect, with their UID, and exits 1.
#
# Reading another user's maps needs root, so on VM24 run it as root (see the
# run sheet). As odbadmin it cannot pass while other users' processes run.
#
# --inspectable-only checks only that every live process can be inspected,
# ignoring py311 use: the preflight that root access works before any
# service stops.
# PREFIX defaults to /home/odbadmin/.pyenv/versions/py311/; --proc exists for
# tests.
set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

usage() { sed -n '2,22p' "$0" >&2; exit 2; }
PROC=/proc
PREFIX=/home/odbadmin/.pyenv/versions/py311/
ONLY=no
while [ $# -gt 0 ]; do
  case "$1" in
    --inspectable-only) ONLY=yes; shift ;;
    --proc) PROC=${2:?--proc needs a value}; shift 2 ;;
    --prefix) PREFIX=${2:?--prefix needs a value}; shift 2 ;;
    *) usage ;;
  esac
done
[ -d "$PROC" ] || { echo "FAIL: $PROC is not a directory" >&2; exit 1; }

# The prefix travels in the environment, not on a command line, so the
# checker's own process never matches it.
P311=$PREFIX PROC_ROOT=$PROC ONLY=$ONLY python3 -I - <<'PY'
import os, sys
p311, root, only = os.environ["P311"], os.environ["PROC_ROOT"], os.environ["ONLY"] == "yes"
me = str(os.getpid())

def gone(pid):
    return not os.path.exists(os.path.join(root, pid))

def uid(pid):
    try:
        with open(os.path.join(root, pid, "status")) as f:
            for line in f:
                if line.startswith("Uid:"):
                    return line.split()[1]
    except OSError:
        pass
    return "?"

def read(pid, name, binary):
    """(text, None) or (None, error name)."""
    try:
        with open(os.path.join(root, pid, name), "rb" if binary else "r",
                  **({} if binary else {"errors": "replace"})) as f:
            data = f.read()
    except OSError as exc:
        return None, exc.__class__.__name__
    if binary:
        data = data.replace(b"\0", b" ").decode(errors="replace").strip()
    return data, None

hits, blind, checked = [], [], 0
for pid in sorted(os.listdir(root), key=lambda p: (len(p), p)):
    if not pid.isdigit() or pid == me:
        continue
    args, err = read(pid, "cmdline", True)
    maps, err2 = (None, None) if err else read(pid, "maps", False)
    if err or err2:
        if gone(pid):
            continue                  # exited meanwhile: nothing to check
        blind.append("%s uid=%s %s (%s)" % (pid, uid(pid), (args or "")[:120], err or err2))
        continue
    checked += 1
    if not only and (p311 in args or p311 in maps):
        hits.append("%s uid=%s %s" % (pid, uid(pid), args[:160]))
if hits:
    print("FAIL: processes use py311:\n  " + "\n  ".join(hits), file=sys.stderr)
if blind:
    print("FAIL: %d live processes could not be inspected, so they may use py311 "
          "(run as root):\n  %s" % (len(blind), "\n  ".join(blind[:50])), file=sys.stderr)
if hits or blind:
    sys.exit(1)
if only:
    print("OK inspectable: all %d live processes could be read" % checked)
else:
    print("OK quiescent: none of %d live processes runs or maps a file from %s" % (checked, p311))
PY
