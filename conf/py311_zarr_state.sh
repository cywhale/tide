#!/bin/bash
# Read-only: report whether the shared py311 is exactly in one of the two
# known states of the Zarr maintenance, using the snapshots that
# py311_zarr_align.sh prepare wrote to DIR.
#
#   py311_zarr_state.sh --dir DIR [--expect before|after] [--python PATH]
#
# Prints "STATE before" (zarr 2.18.6) or "STATE after" (zarr 2.18.7) when all
# of these hold, and otherwise "STATE unknown" with the reasons. It exits 0
# only for a known state, and with --expect only for that state:
#   - zarr imports with the expected version;
#   - the sorted `pip freeze --all` equals DIR/freeze-before.txt (before), or
#     equals it with only the zarr line changed to 2.18.7 (after);
#   - `pip check` stdout and exit status equal DIR/pip-check-before.txt, with
#     nothing on stderr;
#   - exactly one zarr dist-info exists, every file in its RECORD exists with
#     the recorded sha256, and the zarr package directory holds no file
#     outside RECORD (a half-replaced package fails here even when the
#     version and freeze look right).
# It installs nothing and writes nothing: Python runs with -I -B.
set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

FROM=2.18.6
TO=2.18.7

usage() { sed -n '2,20p' "$0" >&2; exit 2; }
DIR=""
EXPECT=""
PY=/home/odbadmin/.pyenv/versions/py311/bin/python
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR=${2:?--dir needs a value}; shift 2 ;;
    --python) PY=${2:?--python needs a value}; shift 2 ;;
    --expect) EXPECT=${2:?--expect needs a value}; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$DIR" ] || usage
case "$EXPECT" in ""|before|after) ;; *) usage ;; esac

reasons=()
unknown() { reasons+=("$1"); }
finish() {
  if [ ${#reasons[@]} -eq 0 ] && [ -n "$1" ]; then
    echo "STATE $1"
    if [ -n "$EXPECT" ] && [ "$1" != "$EXPECT" ]; then
      echo "FAIL: py311 is in the $1 state, expected $EXPECT" >&2
      exit 1
    fi
    exit 0
  fi
  echo "STATE unknown"
  for r in ${reasons[@]+"${reasons[@]}"}; do echo "  - $r"; done
  exit 1
}

[ -x "$PY" ] || { unknown "python not executable: $PY"; finish ""; }
for f in freeze-before.txt pip-check-before.txt; do
  [ -s "$DIR/$f" ] || { unknown "$DIR/$f is missing or empty"; finish ""; }
done

v=$("$PY" -I -B -c 'import zarr; print(zarr.__version__)' 2>/dev/null) \
  || { unknown "zarr does not import"; v=""; }
case "$v" in
  "$FROM") state=before ;;
  "$TO") state=after ;;
  *) state=""; [ -z "$v" ] || unknown "zarr imports as $v" ;;
esac

freeze=$("$PY" -I -B -m pip --disable-pip-version-check freeze --all 2>/dev/null | LC_ALL=C sort -f) \
  || unknown "pip freeze failed"
expect_before=$(cat "$DIR/freeze-before.txt")
expect_after=$(sed "s/^zarr==$FROM\$/zarr==$TO/" "$DIR/freeze-before.txt")
if [ "$freeze" = "$expect_before" ]; then
  [ "$state" = before ] || unknown "freeze matches the before snapshot but zarr imports as '$v'"
elif [ "$freeze" = "$expect_after" ]; then
  [ "$state" = after ] || unknown "freeze matches the after state but zarr imports as '$v'"
else
  unknown "freeze matches neither state: $(diff <(echo "$expect_before") <(echo "$freeze") | grep '^[<>]' | tr '\n' ' ')"
fi

check=$(python3 -I - "$PY" <<'PY'
import subprocess, sys
r = subprocess.run([sys.argv[1], "-I", "-B", "-m", "pip", "--disable-pip-version-check", "check"],
                   capture_output=True, text=True)
if r.stderr:
    print("STDERR " + r.stderr.strip().splitlines()[-1])
else:
    sys.stdout.write(r.stdout + "exit %d" % r.returncode)
PY
) || check="(pip check could not run)"
[ "$check" = "$(cat "$DIR/pip-check-before.txt")" ] \
  || unknown "pip check differs from $DIR/pip-check-before.txt: $(head -c 300 <<< "$check")"

files=$("$PY" -I -B - 2>&1 <<'PY'
import base64, hashlib, importlib.metadata as md, pathlib, sys
# Find zarr's dist-info by name rather than by reading every distribution's
# metadata: py311 holds dist-info directories with only REQUESTED, which
# pip ignores. A zarr one without METADATA does not count as installed.
dirs = sorted({p.resolve() for e in sys.path if e and pathlib.Path(e).is_dir()
               for p in pathlib.Path(e).glob("zarr-*.dist-info")})
real = [p for p in dirs if (p / "METADATA").is_file()]
if len(real) != 1:
    sys.exit("expected one zarr dist-info with METADATA, found %d: %s"
             % (len(real), " ".join(p.name for p in real)))
d = md.Distribution.at(real[0])
listed, bad = set(), []
for f in d.files or []:
    p = pathlib.Path(d.locate_file(f)).resolve()
    listed.add(p)
    if f.hash is None:
        continue
    if not p.is_file():
        bad.append("missing " + str(f))
        continue
    got = base64.urlsafe_b64encode(hashlib.new(f.hash.mode, p.read_bytes()).digest()).rstrip(b"=").decode()
    if got != f.hash.value:
        bad.append("hash mismatch " + str(f))
pkg = pathlib.Path(d.locate_file("zarr")).resolve()
if not pkg.is_dir():
    bad.append("no zarr package directory")
else:
    for p in pkg.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and p.resolve() not in listed:
            bad.append("not in RECORD " + str(p.relative_to(pkg.parent)))
if not d.files:
    bad.append("zarr has no RECORD")
if bad:
    sys.exit("; ".join(bad[:10]) + (" (+%d more)" % (len(bad) - 10) if len(bad) > 10 else ""))
PY
) || unknown "zarr files: $files"

finish "$state"
