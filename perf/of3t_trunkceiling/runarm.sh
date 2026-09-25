#!/usr/bin/env bash
# of3t-trunkceiling: one trunk-gradient arm at crop 384, with a named lever set.
#   runarm.sh <TAG> <lever> [KEY=VAL ...] [-- extra dev_grad args]
# Wall-clock is irrelevant to this row and qb2's AICLK is clamped at 800 MHz today, so the
# sampler runs to RECORD the clock, not to defend a timing. An accuracy reading does not move
# with the clock; nothing here is quoted as a perf number.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkceiling
cd "$W"
O=/tmp/of3t/of3t-trunkceiling
mkdir -p "$O"
B=/home/ttuser/of3t_trunk043ref/boundary_n384.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt

TAG=$1; LEVER=$2; shift 2
ENVS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")

OUT=$O/dev_${TAG}.pt
REP=$W/perf/of3t_trunkceiling/DEV_${TAG}.json
CEN=$W/perf/of3t_trunkceiling/CENSUS_${TAG}.json
CLK=$O/aiclk_${TAG}.txt

: > "$CLK"
( while true; do
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["device_info"][0]["telemetry"]["aiclk"].strip())' \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!

S=$(date +%s)
echo "=== $TAG lever=$LEVER start $(date -u +%FT%TZ) qb2 card 0 env=${ENVS[*]:-none} ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
env "${ENVS[@]}" \
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-trunkceiling \
  OMP_NUM_THREADS=8 PYTHONPATH="$W" \
  timeout 3000 python3 perf/of3t_trunkceiling/arm.py --census-out "$CEN" \
    --lever "$LEVER" -- \
    --boundary "$B" --cap-last "$C" --out "$OUT" \
    --report "$REP" --arm flipped --crop 0 "${EXTRA[@]}" 2>&1 \
  | grep -E '^\{|^CENSUS|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed|config ' | tail -30
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card 0, p300c Blackhole, qb2, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
ls -la "$OUT" 2>/dev/null
exit $rc
