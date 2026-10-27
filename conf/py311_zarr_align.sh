#!/bin/bash
# Align the shared VM24 py311 Zarr from 2.18.6 to the uv.lock pin 2.18.7, or
# roll it back, changing no other package.
#
#   py311_zarr_align.sh prepare  --dir NEW_DIR [--python PATH]
#   py311_zarr_align.sh apply    --dir DIR [--python PATH]
#   py311_zarr_align.sh rollback --dir DIR [--python PATH]
#
# prepare changes nothing in py311. DIR must not exist yet. It records the
#   sorted `pip freeze --all`, `pip check`, and the Zarr version, downloads
#   both hash-pinned wheels into DIR/wheels, and dry-runs the install, which
#   must offer exactly zarr 2.18.7.
# apply refuses unless prepare completed and py311 still matches its freeze.
#   It installs zarr 2.18.7 offline, hash-pinned, with --no-deps. It then
#   requires the freeze to differ from prepare's only in the zarr line,
#   `pip check` to be unchanged, and zarr to import as 2.18.7.
# rollback installs 2.18.6 the same way and requires the freeze and
#   `pip check` to equal prepare's again.
#
# Fails closed: any failed step exits non-zero before the next one, so a
# restart or smoke chained after it with && never runs. Snapshots go to a
# temporary name and are renamed only when pip succeeded and listed zarr, so
# a failed snapshot never leaves a valid-looking file. apply and rollback
# change the live py311: Gunicorn apps on it that run with --reload (Tide and
# MHW on VM24) reload when the zarr files change.
set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

FROM=2.18.6
TO=2.18.7
HASH_FROM=a5334311fae8859870d78561ebb6bd0ab01accc392a6cf4dcaa25d83b4a2f22c
HASH_TO=ac3dc4033e9ae4e9d7b5e27c97ea3eaf1003cc0a07f010bd83d5134bf8c4b223

die() { echo "FAIL: $*" >&2; exit 1; }
usage() { sed -n '2,25p' "$0" >&2; exit 2; }

pip_() { "$PY" -I -m pip --disable-pip-version-check "$@"; }

zarr_version() {
  "$PY" -I -c 'import zarr; print(zarr.__version__)' \
    || die "cannot import zarr with $PY"
}

# snapshot NAME: sorted `pip freeze --all` into DIR/NAME. Written to a
# temporary file and renamed only if pip succeeded and listed zarr; a failed
# snapshot leaves no DIR/NAME.
snapshot() {
  local out="$DIR/$1" raw="$DIR/.$1.partial"
  if ! pip_ freeze --all > "$raw" 2> "$out.stderr"; then
    rm -f "$raw"
    die "pip freeze failed (see $out.stderr); no $1 was written"
  fi
  if ! grep -q '^zarr==' "$raw"; then
    rm -f "$raw"
    die "pip freeze listed no zarr; no $1 was written"
  fi
  LC_ALL=C sort -f "$raw" > "$out.tmp" || { rm -f "$raw" "$out.tmp"; die "cannot sort the freeze"; }
  rm -f "$raw"
  mv "$out.tmp" "$out"
}

# pip_check NAME: `pip check` output plus its exit status into DIR/NAME. Exit
# 1 means conflicts were found, which existing py311 conflicts make normal;
# anything else means pip itself failed.
pip_check() {
  local out="$DIR/$1" rc=0
  pip_ check > "$out.tmp" 2>&1 || rc=$?
  [ "$rc" -le 1 ] || { rm -f "$out.tmp"; die "pip check failed to run (exit $rc)"; }
  echo "exit $rc" >> "$out.tmp"
  mv "$out.tmp" "$out"
}

# install VERSION LOG: offline, hash-pinned, no dependency changes. pipefail
# makes a pip failure fail the pipeline even though tee succeeds.
install() {
  pip_ install --no-deps --no-index --find-links "$DIR/wheels" --require-hashes \
    -r "$DIR/zarr-$1.req" 2>&1 | tee "$DIR/$2"
}

cmd_prepare() {
  local v
  [ ! -e "$DIR" ] || die "$DIR already exists; use a new directory"
  mkdir -p "$(dirname "$DIR")"
  mkdir "$DIR"
  v=$(zarr_version)
  [ "$v" = "$FROM" ] || die "py311 has zarr $v, expected $FROM"
  snapshot freeze-before.txt
  grep -qx "zarr==$FROM" "$DIR/freeze-before.txt" \
    || die "freeze does not list zarr==$FROM"
  pip_check pip-check-before.txt
  echo "zarr==$TO --hash=sha256:$HASH_TO" > "$DIR/zarr-$TO.req"
  echo "zarr==$FROM --hash=sha256:$HASH_FROM" > "$DIR/zarr-$FROM.req"
  for v in "$TO" "$FROM"; do
    pip_ download -q --no-deps --require-hashes -r "$DIR/zarr-$v.req" \
      -d "$DIR/wheels" || die "cannot download the hash-pinned zarr $v wheel"
    [ -f "$DIR/wheels/zarr-$v-py3-none-any.whl" ] || die "zarr $v wheel missing"
  done
  pip_ install --dry-run --no-deps --no-index --find-links "$DIR/wheels" \
    --require-hashes -r "$DIR/zarr-$TO.req" > "$DIR/dry-run.txt" 2>&1 \
    || die "dry run failed (see $DIR/dry-run.txt)"
  [ "$(grep '^Would install' "$DIR/dry-run.txt")" = "Would install zarr-$TO" ] \
    || die "dry run does not offer exactly zarr-$TO (see $DIR/dry-run.txt)"
  echo "prepared $(date -u +%FT%TZ) $PY zarr $FROM" > "$DIR/prepared"
  echo "OK prepare: py311 unchanged; snapshots and wheels in $DIR"
}

cmd_apply() {
  local v changed
  [ -f "$DIR/prepared" ] || die "$DIR was not prepared; run prepare first"
  [ ! -e "$DIR/applied" ] || die "$DIR was already used for apply"
  v=$(zarr_version)
  [ "$v" = "$FROM" ] || die "py311 has zarr $v, expected $FROM; nothing was installed"
  snapshot freeze-pre-apply.txt
  cmp -s "$DIR/freeze-before.txt" "$DIR/freeze-pre-apply.txt" \
    || die "py311 changed since prepare; nothing was installed"
  : > "$DIR/applied"
  install "$TO" install.log \
    || die "pip install failed; py311 state is unknown. Do not restart or smoke; inspect it, then roll back only on the owner's decision"
  snapshot freeze-after.txt
  changed=$(diff "$DIR/freeze-before.txt" "$DIR/freeze-after.txt" | grep '^[<>]' || true)
  [ "$changed" = "$(printf '< zarr==%s\n> zarr==%s' "$FROM" "$TO")" ] \
    || die "freeze changed beyond zarr: $changed; do not restart or smoke"
  pip_check pip-check-after.txt
  cmp -s "$DIR/pip-check-before.txt" "$DIR/pip-check-after.txt" \
    || die "pip check changed (see $DIR/pip-check-*.txt); do not restart or smoke"
  v=$(zarr_version)
  [ "$v" = "$TO" ] || die "zarr imports as $v, expected $TO; do not restart or smoke"
  echo "OK apply: zarr $FROM -> $TO, no other package changed, pip check unchanged"
}

cmd_rollback() {
  local v
  [ -f "$DIR/prepared" ] || die "$DIR was not prepared; nothing to roll back to"
  install "$FROM" rollback.log \
    || die "pip install of zarr $FROM failed; py311 state is unknown. Do not restart or smoke"
  snapshot freeze-rollback.txt
  cmp -s "$DIR/freeze-before.txt" "$DIR/freeze-rollback.txt" \
    || die "freeze differs from the prepare snapshot; do not restart or smoke"
  pip_check pip-check-rollback.txt
  cmp -s "$DIR/pip-check-before.txt" "$DIR/pip-check-rollback.txt" \
    || die "pip check differs from the prepare snapshot; do not restart or smoke"
  v=$(zarr_version)
  [ "$v" = "$FROM" ] || die "zarr imports as $v, expected $FROM"
  echo "OK rollback: zarr back to $FROM; freeze and pip check match the prepare snapshot"
}

[ $# -ge 1 ] || usage
MODE=$1
shift
DIR=""
PY=/home/odbadmin/.pyenv/versions/py311/bin/python
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR=${2:?--dir needs a value}; shift 2 ;;
    --python) PY=${2:?--python needs a value}; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$DIR" ] || usage
[ -x "$PY" ] || die "python not executable: $PY"

case "$MODE" in
  prepare) cmd_prepare ;;
  apply) cmd_apply ;;
  rollback) cmd_rollback ;;
  *) usage ;;
esac
