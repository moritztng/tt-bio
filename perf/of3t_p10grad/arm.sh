#!/usr/bin/env bash
# of3t-p10grad: the per-op float64 gradient sweep. qb2 (tt-quietbox2), card 0.
#   arm.sh <TAG> [extra opgrad.py args]
# AICLK is sampled every 4 s DURING the arm from a separate shell, and the harness samples it
# again in-process between cases. The reading here is an ACCURACY reading, so the clock does not
# move it; it is recorded because a number without a DURING-sampled clock is not a measurement
# on this hardware.
set -uo pipefail
# The worktree this script is IN, not a hard-coded one: of3t-p10lnbw runs the same arms
# against a patched closure and a path pinned to another worktree would have graded main.
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
CARD=${PG_CARD:-0}
SLUG=${PG_SLUG:-of3t-p10grad}
TAG=${1:?tag}; shift

O=/tmp/of3t/$SLUG
mkdir -p "$O"
REP=$W/perf/of3t_p10grad/PEROP_${TAG}.json
CLK=$O/aiclk_${TAG}.txt
LOG=$O/${TAG}.log

: > "$CLK"
( while true; do
    TT_VISIBLE_DEVICES=$CARD /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][0]['telemetry']['aiclk'].strip())" \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

S=$(date +%s)
echo "=== $SLUG $TAG start $(date -u +%FT%TZ) host=$(hostname) card=$CARD head=$(git rev-parse --short HEAD) ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
TT_BIO_LEASE_HOLDER=worker:$SLUG OMP_NUM_THREADS=8 \
python3 perf/of3t_p10grad/opgrad.py --out "$REP" --card "$CARD" "$@" 2>&1 \
  | grep -vE '^\s*$|DEBUG|^Config\{' | tail -60
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $SLUG $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card $CARD, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
exit $rc
