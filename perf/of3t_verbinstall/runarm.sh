#!/usr/bin/env bash
# of3t-verbinstall: one trunk-gradient arm at crop 384 on qb1 card 1, softmax from the package.
#   runarm.sh <TAG> <lever> [KEY=VAL ...] [-- extra dev_grad args]
# The clock is RECORDED, not defended: an accuracy reading does not move with AICLK and nothing
# here is quoted as a perf number. The sampler runs anyway because a run that records no clock
# cannot be compared with one that does.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-verbinstall
cd "$W"
O=/tmp/of3t/of3t-verbinstall
mkdir -p "$O"
# The paths of3t-trunkceiling used are gone with its worktree; these are the surviving copies
# and their sha256 match DEV_CEIL_HF3.json's recorded boundary_sha256 / cotangent_sha256
# exactly (8cb3a586... and a55ef1c4...), so this is the same frame and not a lookalike.
B=/home/ttuser/of3t_frame384/boundary_n384.pt
C=/home/ttuser/of3t_frame384/block47_boundary.pt

TAG=$1; LEVER=$2; shift 2
ENVS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")

OUT=$O/dev_${TAG}.pt
REP=$W/perf/of3t_verbinstall/DEV_${TAG}.json
CEN=$W/perf/of3t_verbinstall/CENSUS_${TAG}.json
SMX=$W/perf/of3t_verbinstall/EXACT_SOFTMAX_${TAG}.json
CLK=$O/aiclk_${TAG}.txt

: > "$CLK"
( while true; do
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["device_info"][1]["telemetry"]["aiclk"].strip())' \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!

S=$(date +%s)
echo "=== $TAG lever=$LEVER start $(date -u +%FT%TZ) qb1 card 1 env=${ENVS[*]:-none} ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
env "${ENVS[@]}" \
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:of3t-verbinstall \
  OMP_NUM_THREADS=8 PYTHONPATH="$W" \
  timeout ${ARM_TIMEOUT:-7200} python3 perf/of3t_verbinstall/pkgarm.py \
    --exact-softmax-out "$SMX" --census-out "$CEN" \
    --lever "$LEVER" -- \
    --boundary "$B" --cap-last "$C" --out "$OUT" \
    --report "$REP" --arm flipped --crop 0 "${EXTRA[@]}" 2>&1 \
  | grep -E '^\{|^CENSUS|^EXACT_SOFTMAX|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed|config ' | tail -30
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card 1, p150a Blackhole, qb1, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
ls -la "$OUT" 2>/dev/null
exit $rc
