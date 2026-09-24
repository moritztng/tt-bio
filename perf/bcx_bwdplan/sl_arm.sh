#!/usr/bin/env bash
# OF3T's SL arm on qb1 (SL_CARD, default 0): bcx-heads' `sl_arm.sh`, the argv, env and inputs of
# of3t-stackexact's `arm.sh`, with every path moved to this row's own worktree and tmpfs.
#
#   sl_arm.sh <TAG>      run from the OF3T test tree the arm is to test
set -uo pipefail
TAG=${1:?tag}; CARD=${SL_CARD:-0}; NODE=${SL_AICLK_NODE:-1}      # qb1: card 0 is sysfs node 1, card 3 node 0
W=$(pwd)
BP=/home/ttuser/.coworker/wt/bcx-bwdplan
I=/dev/shm/bcx-bp-of3t-inputs
O=/dev/shm/bcx-bp-sl; mkdir -p "$O"
R=$BP/perf/bcx_bwdplan/sl; mkdir -p "$R"
B=$I/boundary_model_n384.pt; C=$I/cot_external.pt
[ "$(sha256sum < "$B" | cut -d' ' -f1)" = 583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2 ] || { echo "boundary digest mismatch"; exit 2; }
[ "$(sha256sum < "$C" | cut -d' ' -f1)" = 1d15a8dc6db2a14b678e5ed5d92558af99d369599226391fa59f36ed84192ef4 ] || { echo "cotangent digest mismatch (A42)"; exit 2; }
unset TT_MESH_GRAPH_DESC_PATH
source /home/ttuser/tt-bio-dev/env/bin/activate
export TMPDIR=/dev/shm/bcx-bp-tmp TT_METAL_CACHE=/dev/shm/bcx-bp-cache
ENV=(TT_BIO_SOFTMAX_BW_RENORM=1 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
     TT_BIO_LEASE_HOLDER=worker:bcx-bwdplan OMP_NUM_THREADS=8 PYTHONPATH="$W")
CLK=$R/aiclk_${TAG}.txt; : > "$CLK"
( while true; do
    cat "/sys/class/tenstorrent/tenstorrent!${NODE}/tt_aiclk" >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
echo "=== $TAG start $(date -u +%FT%TZ) $(hostname) card $CARD node $NODE commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain tt_bio | wc -l) load $(cut -d' ' -f1-3 /proc/loadavg)"
S=$(date +%s)
env "${ENV[@]}" timeout ${SL_TIMEOUT:-5400} python3 perf/of3t_stackexact/stackarm.py --exact softmax,layer_norm \
    --stats-out "$R/STACK_EXACT_${TAG}.json" --census-out "$R/CENSUS_${TAG}.json" --lever none -- \
    --boundary "$B" --cap-last "$C" --out "$O/dev_${TAG}.pt" --report "$R/DEV_${TAG}.json" --arm flipped --crop 0 2>&1 \
  | tee "$O/raw_${TAG}.log" | grep --line-buffered -E '^\{|OUR GRADIENT|STACK_EXACT|Traceback|rror|FAILED' | cut -c1-400 | tail -25
rc=${PIPESTATUS[0]}
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $(( $(date +%s) - S ))s $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
python3 -c "import sys;s=sorted(int(l) for l in open('$CLK') if l.strip().isdigit());print('AICLK DURING n',len(s),'min',s[0],'median',s[len(s)//2],'max',s[-1]) if s else print('no clock')"
if [ -s "$O/dev_${TAG}.pt" ]; then
  sha256sum "$O/dev_${TAG}.pt" | tee "$R/dev_${TAG}.sha256"
  python3 "$BP/perf/bcx_heads/ptdigest.py" "$O/dev_${TAG}.pt" "$R/digest_${TAG}.json" && rm -f "$O/dev_${TAG}.pt"
  for ref in "$BP/perf/bcx_heads/sl/digest_banked_SL.json" "$BP/perf/bcx_heads/sl/digest_SL_POSTG.json"; do
    python3 "$BP/perf/bcx_heads/ptdigest.py" --cmp "$ref" "$R/digest_${TAG}.json"
  done | tee "$R/cmp_${TAG}.jsonl"
fi
exit "$rc"
