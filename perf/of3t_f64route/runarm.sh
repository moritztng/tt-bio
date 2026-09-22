#!/usr/bin/env bash
# of3t-f64route: one trunk-gradient arm at crop 384 on qb1, with a named lever set.
#   runarm.sh <TAG> <lever> <card> [KEY=VAL ...] [-- extra dev_grad args]
# Same shape as of3t-trunkceiling/runarm.sh, pointed at qb1's copy of the boundary pair and at
# this worktree. Accuracy does not move with the clock; the sampler records it, it does not
# defend a timing.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-f64route
cd "$W"
O=/tmp/of3t/of3t-f64route
mkdir -p "$O"
B=/home/ttuser/of3t_frame384/boundary_n384.pt
C=/home/ttuser/of3t_frame384/block47_boundary.pt

TAG=$1; LEVER=$2; CARD=$3; shift 3
ENVS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")

OUT=$O/dev_${TAG}.pt
REP=$W/perf/of3t_f64route/DEV_${TAG}.json
CEN=$W/perf/of3t_f64route/CENSUS_${TAG}.json
CLK=$O/aiclk_${TAG}.txt

: > "$CLK"
export CARD
( while true; do
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c 'import sys,json,os;d=json.load(sys.stdin);print(d["device_info"][int(os.environ["CARD"])]["telemetry"]["aiclk"].strip())' \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!

S=$(date +%s)
echo "=== $TAG lever=$LEVER start $(date -u +%FT%TZ) qb1 card $CARD env=${ENVS[*]:-none} ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
env "${ENVS[@]}" \
  TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-f64route \
  OMP_NUM_THREADS=8 PYTHONPATH="$W" \
  timeout 5400 python3 perf/of3t_f64route/arm.py --census-out "$CEN" \
    --lever "$LEVER" -- \
    --boundary "$B" --cap-last "$C" --out "$OUT" \
    --report "$REP" --arm flipped --crop 0 "${EXTRA[@]}" 2>&1 \
  | grep -E '^\{|^CENSUS|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed|config ' | tail -30
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (tt-smi index 0 == granted card, p150a Blackhole, qb1, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
ls -la "$OUT" 2>/dev/null
exit $rc
