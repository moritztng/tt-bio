#!/usr/bin/env bash
# Attribute the perf leg's -16.2 % on protenix-v2. Three arms, same harness, same box, same hour:
#   off   — bucket disabled. THE control. The committed baseline (1.472 str/s) is historical, so
#           without this arm any part of the delta could be box drift rather than the flip
#           (perf-page-cell-is-historical-not-live-baseline).
#   m32   — bucket on at pad multiple 32, so 20 aa -> 32 instead of 64.
#   m64   — bucket on at the shipped default. Already measured at 1.234; re-run here so all three
#           arms come from one box state.
set -u
: "${CARD:?set CARD}"
WT=/home/ttuser/.coworker/wt/protenix-opendde-token-bucket-flip-measure
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:protenix-opendde-token-bucket-flip-measure"
OUT=perf/tokenbucket/padsweep
mkdir -p $OUT

arm() {  # arm <tag> <env assignments...>
  tag=$1; shift
  [ -f "$OUT/$tag.log" ] && { echo "SKIP $tag"; return 0; }
  echo "=== $(date -Is) $tag ($*) load $(cut -d' ' -f1 /proc/loadavg)"
  env $LEASE PYTHONPATH=$WT ESM_ROOT=$ESM_ROOT "$@" \
    "$PY" -u scripts/perf_regression.py --model protenix-v2 > "$OUT/$tag.log" 2>&1
  grep -aE "structures/s\s+[0-9]" "$OUT/$tag.log" | tail -1
}

arm off TT_BIO_PROTENIX_TOKEN_BUCKET=0
arm m32 TT_BIO_PROTENIX_TOKEN_BUCKET=1 TT_BIO_PROTENIX_TOKEN_PAD_MULTIPLE=32
arm m64 TT_BIO_PROTENIX_TOKEN_BUCKET=1 TT_BIO_PROTENIX_TOKEN_PAD_MULTIPLE=64
echo "=== $(date -Is) padsweep done"
