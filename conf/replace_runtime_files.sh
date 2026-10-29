#!/bin/bash
# Replace the v0.3.2 runtime manifest in a release directory file by file,
# or restore it from the backups taken by an earlier deploy.
#
#   replace_runtime_files.sh deploy  --rel DIR --src CHECKOUT --commit SHA --work NEW_DIR
#   replace_runtime_files.sh restore --rel DIR --work DEPLOY_WORK_DIR
#
# deploy: SRC must be a checkout at COMMIT. The expected hash of each file is
#   its blob at that commit. WORK must not exist yet (earlier backups are never
#   overwritten) and must be on the release filesystem.
# restore: puts back the backups recorded in WORK/backup.sha256, which must
#   list exactly the manifest paths (see load_backup_record).
#
# Each file is staged inside WORK and verified against its expected SHA-256.
# All files are staged and verified, and then every staged copy is verified
# again, before the first replacement: a mismatch replaces nothing. Each copy
# is verified once more right before its own rename(2) over the target, so the
# --reload Gunicorn never sees a partial file, and the target is verified after
# it. Each rename is atomic, but the replacements together are not one
# transaction: a failure after the first rename leaves the files renamed so
# far in place, and the error names the restore command to run before
# anything else. Any failure exits non-zero.
set -euo pipefail
# bash >= 4.4: keep errexit inside $(...); critical steps also check explicitly.
shopt -s inherit_errexit 2>/dev/null || true

MANIFEST=(conf/start_app.sh src/dask_client_manager.py)

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,22p' "$0" >&2; exit 2; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
sha256_stdin() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1
  else shasum -a 256 | cut -d' ' -f1; fi
}
# GNU stat first (VM24), BSD stat as fallback (macOS test runs).
dev_of() { stat -c %d "$1" 2>/dev/null || stat -f %d "$1"; }
mode_of() { stat -c %a "$1" 2>/dev/null || stat -f %Lp "$1"; }

same_fs() {
  local first p d
  first=$(dev_of "$1")
  for p in "$@"; do
    d=$(dev_of "$p")
    [ "$d" = "$first" ] || die "$p is on device $d, not $first; rename would not be atomic"
  done
}

runtime_hashes() {  # run inside the release directory
  local f
  for f in tide_app.py __init__.py src/*.py conf/start_app.sh conf/ecosystem.config.js; do
    if [ -f "$f" ]; then echo "$(sha256_file "$f")  $f"; fi
  done
}

# Load WORK/backup.sha256 into BACKUP_HASH (indexed like MANIFEST). The record
# must hold exactly one well-formed "<sha256>  <path>" line per manifest path
# and nothing else, so a truncated, duplicated, or tampered record (including
# paths outside the manifest, such as ../ traversal) is refused.
load_backup_record() {
  local re='^([0-9a-f]{64})  (.+)$' line h f i n=0 matched
  BACKUP_HASH=()
  for i in "${!MANIFEST[@]}"; do BACKUP_HASH[$i]=""; done
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    [[ $line =~ $re ]] || die "malformed line $n in backup record: $line"
    h=${BASH_REMATCH[1]}
    f=${BASH_REMATCH[2]}
    matched=""
    for i in "${!MANIFEST[@]}"; do
      if [ "${MANIFEST[$i]}" = "$f" ]; then
        [ -z "${BACKUP_HASH[$i]}" ] || die "duplicate path in backup record: $f"
        BACKUP_HASH[$i]=$h
        matched=yes
      fi
    done
    [ -n "$matched" ] || die "unexpected path in backup record: $f"
  done < "$1"
  for i in "${!MANIFEST[@]}"; do
    [ -n "${BACKUP_HASH[$i]}" ] || die "backup record is missing ${MANIFEST[$i]}"
  done
}

is_manifest() {
  local m
  for m in "${MANIFEST[@]}"; do [ "$m" = "$1" ] && return 0; done
  return 1
}

# stage SOURCE EXPECTED_SHA TARGET DIR: copy SOURCE into DIR with TARGET's
# mode, verify it, and print the staged path.
stage() {
  local src=$1 expected=$2 target=$3 dir=$4 staged
  [ -f "$src" ] || die "missing source: $src"
  [ -f "$target" ] || die "missing target: $target"
  staged="$dir/$(echo "$target" | tr '/' '_')"
  cp "$src" "$staged" || die "cannot stage $src"
  chmod "$(mode_of "$target")" "$staged" || die "cannot set mode on $staged"
  [ "$(sha256_file "$staged")" = "$expected" ] \
    || die "hash mismatch for $target (staged from $src); nothing was replaced"
  echo "$staged"
}

# recheck_staged ENTRY...: each "STAGED|EXPECTED_SHA|PATH" copy still matches,
# immediately before the first rename. A staged copy can change after its own
# check while later files are staged; this keeps such a copy out of REL.
recheck_staged() {
  local entry s h f
  for entry in "$@"; do
    IFS='|' read -r s h f <<< "$entry"
    [ "$(sha256_file "$s")" = "$h" ] \
      || die "staged copy of $f changed after verification; nothing was replaced"
  done
}

# swap STAGED EXPECTED_SHA TARGET: re-verify the staged copy, rename it over
# TARGET (atomic for this one file), then re-verify TARGET.
swap() {
  [ "$(sha256_file "$1")" = "$2" ] \
    || die "staged copy for $3 changed before its rename; $3 was not replaced, but earlier files may have been; run '$0 restore --rel $REL --work $WORK'"
  mv -f "$1" "$3" || die "rename failed for $3; run '$0 restore --rel $REL --work $WORK'"
  [ "$(sha256_file "$3")" = "$2" ] \
    || die "post-rename hash mismatch for $3; run '$0 restore --rel $REL --work $WORK'"
}

cmd_deploy() {
  local head f h s expected entry staged_list
  [ -n "$REL" ] && [ -n "$SRC" ] && [ -n "$COMMIT" ] && [ -n "$WORK" ] || usage
  [ -d "$REL" ] || die "release directory not found: $REL"
  [ "${#COMMIT}" -ge 7 ] || die "--commit needs at least 7 hex digits"
  [ ! -e "$WORK" ] || die "work directory already exists: $WORK (refusing to overwrite earlier backups)"
  head=$(git -C "$SRC" rev-parse HEAD)
  case "$head" in
    "$COMMIT"*) ;;
    *) die "source checkout is at $head, not $COMMIT" ;;
  esac

  mkdir -p "$(dirname "$WORK")"
  mkdir "$WORK" "$WORK/backup" "$WORK/staging"
  same_fs "$WORK/staging" "$REL/src" "$REL/conf"
  echo "$head" > "$WORK/source-commit"
  (cd "$REL" && runtime_hashes) > "$WORK/runtime-before.sha256"

  : > "$WORK/backup.sha256"
  for f in "${MANIFEST[@]}"; do
    mkdir -p "$WORK/backup/$(dirname "$f")"
    cp -p "$REL/$f" "$WORK/backup/$f"
    h=$(sha256_file "$REL/$f")
    [ "$(sha256_file "$WORK/backup/$f")" = "$h" ] || die "backup copy mismatch: $f"
    echo "$h  $f" >> "$WORK/backup.sha256"
  done

  staged_list=()
  for f in "${MANIFEST[@]}"; do
    expected=$(git -C "$SRC" show "HEAD:$f" | sha256_stdin)
    s=$(stage "$SRC/$f" "$expected" "$REL/$f" "$WORK/staging")
    staged_list+=("$s|$expected|$f")
  done
  recheck_staged "${staged_list[@]}"

  for entry in "${staged_list[@]}"; do
    IFS='|' read -r s expected f <<< "$entry"
    swap "$s" "$expected" "$REL/$f"
    echo "replaced $f"
  done

  while read -r h f; do
    is_manifest "$f" && continue
    [ "$(sha256_file "$REL/$f")" = "$h" ] || die "runtime file changed unexpectedly: $f"
  done < "$WORK/runtime-before.sha256"
  rmdir "$WORK/staging"
  echo "OK deploy: ${MANIFEST[*]} now match $head; backups in $WORK/backup"
}

cmd_restore() {
  local sdir h f i s entry staged_list
  [ -n "$REL" ] && [ -n "$WORK" ] || usage
  [ -f "$WORK/backup.sha256" ] || die "no backup record in $WORK"
  load_backup_record "$WORK/backup.sha256"
  sdir=$(mktemp -d "$WORK/restore-staging.XXXXXX")
  same_fs "$sdir" "$REL/src" "$REL/conf"

  staged_list=()
  for i in "${!MANIFEST[@]}"; do
    f=${MANIFEST[$i]}
    h=${BACKUP_HASH[$i]}
    s=$(stage "$WORK/backup/$f" "$h" "$REL/$f" "$sdir")
    staged_list+=("$s|$h|$f")
  done
  recheck_staged "${staged_list[@]}"

  for entry in "${staged_list[@]}"; do
    IFS='|' read -r s h f <<< "$entry"
    swap "$s" "$h" "$REL/$f"
    echo "restored $f"
  done
  rmdir "$sdir"
  echo "OK restore: manifest files match the backups in $WORK/backup"
}

[ $# -ge 1 ] || usage
MODE=$1
shift
REL="" SRC="" COMMIT="" WORK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rel) REL=${2:?--rel needs a value}; shift 2 ;;
    --src) SRC=${2:?--src needs a value}; shift 2 ;;
    --commit) COMMIT=${2:?--commit needs a value}; shift 2 ;;
    --work) WORK=${2:?--work needs a value}; shift 2 ;;
    *) usage ;;
  esac
done

case "$MODE" in
  deploy) cmd_deploy ;;
  restore) cmd_restore ;;
  *) usage ;;
esac
