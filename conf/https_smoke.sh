#!/bin/bash
# HTTPS smoke for the Tide API on VM24 through local resolution, with
# certificate verification on (never -k). Exits non-zero unless every check
# passes.
#
#   https_smoke.sh --host HOST --out-root DIR --mode same|different|prod
#                  [--canary-port 8041] [--prod-port 8040] [--query PATH?QUERY]
#
# same:      canary and production must both return 200 with identical bodies
#            (TPXO10 canary).
# different: both must return 200 and the bodies must differ (TPXO9 canary).
# prod:      production alone must return 200 (post-deploy check).
# Every response must be HTTP 200 with a non-empty body other than "{}".
# Each run writes its bodies to a new unique directory under DIR (mktemp -d,
# printed as "attempt:"), so retries never collide and earlier attempts are
# kept. Requires curl >= 7.76 (--fail-with-body); older curl is refused
# before any request.
set -euo pipefail

die() { echo "FAIL: $*" >&2; exit 1; }
usage() { sed -n '2,17p' "$0" >&2; exit 2; }

require_curl_7_76() {
  local ver major minor
  ver=$(curl --version | awk 'NR == 1 {print $2}')   # awk reads all input: no SIGPIPE
  IFS=. read -r major minor _ <<< "$ver"
  case "$major$minor" in
    ''|*[!0-9]*) die "cannot parse curl version '$ver'; need curl >= 7.76" ;;
  esac
  if [ "$major" -lt 7 ] || { [ "$major" -eq 7 ] && [ "$minor" -lt 76 ]; }; then
    die "curl $ver lacks --fail-with-body; need curl >= 7.76"
  fi
}

HOST="" OUT_ROOT="" MODE="" CANARY_PORT=8041 PROD_PORT=8040
QUERY='api/tide?lon0=-157.86453&lat0=21.303333&start=2023-07-25&end=2023-07-26'
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST=${2:?--host needs a value}; shift 2 ;;
    --out-root) OUT_ROOT=${2:?--out-root needs a value}; shift 2 ;;
    --mode) MODE=${2:?--mode needs a value}; shift 2 ;;
    --canary-port) CANARY_PORT=${2:?}; shift 2 ;;
    --prod-port) PROD_PORT=${2:?}; shift 2 ;;
    --query) QUERY=${2:?}; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$HOST" ] && [ -n "$OUT_ROOT" ] || usage
case "$MODE" in same|different|prod) ;; *) usage ;; esac
require_curl_7_76
mkdir -p "$OUT_ROOT"
OUT=$(mktemp -d "$OUT_ROOT/$(date +%Y%m%dT%H%M%S)-$MODE-XXXXXX") \
  || die "cannot create an attempt directory under $OUT_ROOT"
echo "attempt: $OUT"

fetch() {  # PORT FILE
  local code
  code=$(curl -sS --fail-with-body --resolve "$HOST:$1:127.0.0.1" \
           -o "$2" -w '%{http_code}' "https://$HOST:$1/$QUERY") \
    || die "request to port $1 failed (HTTP ${code:-none}); body in $2"
  [ "$code" = "200" ] || die "port $1 returned HTTP $code, expected 200"
  [ -s "$2" ] || die "port $1 returned an empty body"
  [ "$(tr -d ' \t\r\n' < "$2")" != "{}" ] || die "port $1 returned {} (no model data at the query point)"
  echo "port $1: HTTP 200, $(wc -c < "$2" | tr -d ' ') bytes -> $2"
}

if [ "$MODE" = "prod" ]; then
  fetch "$PROD_PORT" "$OUT/prod.json"
  echo "PASS prod: HTTP 200 from port $PROD_PORT"
  exit 0
fi

fetch "$CANARY_PORT" "$OUT/canary.json"
fetch "$PROD_PORT" "$OUT/prod.json"
if cmp -s "$OUT/canary.json" "$OUT/prod.json"; then identical=yes; else identical=no; fi
case "$MODE" in
  same)
    [ "$identical" = "yes" ] || die "canary and production bodies differ (expected identical for the TPXO10 store)" ;;
  different)
    [ "$identical" = "no" ] || die "canary and production bodies are identical (expected a difference for the TPXO9 store)" ;;
esac
echo "PASS $MODE: canary port $CANARY_PORT vs production port $PROD_PORT"
