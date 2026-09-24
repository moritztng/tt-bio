#!/usr/bin/env bash
# OF3T's SL arm (of3t-stackexact arm.sh, TAG SL, scopes softmax,layer_norm) on qb1.
#
#   sl_arm.sh <TAG> <card>     run from the tree the arm is to test
#
# Same boundary and cotangent (digests checked), same stackarm.py argv, env and thread count as
# arm.sh. Two differences, both forced by the host: the inputs are copies under .of3t/inputs,
# and there is no p300c refusal, because qb1 is p150a. So a reading here is compared against
# another reading on this same card, never against the banked qb2 dev_SL.pt as if it were one
# instrument. The .pt is digested per tensor and then deleted (disk).
set -uo pipefail
TAG=${1:?tag}; CARD=${2:?card}
W=$(pwd)
BH=/home/ttuser/.coworker/wt/bcx-heads
I=$BH/.of3t/inputs
O=${SL_OUT:-$BH/.of3t/out}; mkdir -p "$O"   # the .pt is 1.3 GB and qb1 root once filled mid-save: use /dev/shm
R=$BH/perf/bcx_heads/sl; mkdir -p "$R"
B=$I/boundary_model_n384.pt; C=$I/cot_external.pt
[ "$(sha256sum < "$B" | cut -d' ' -f1)" = 583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2 ] || { echo "boundary digest mismatch"; exit 2; }
[ "$(sha256sum < "$C" | cut -d' ' -f1)" = 1d15a8dc6db2a14b678e5ed5d92558af99d369599226391fa59f36ed84192ef4 ] || { echo "cotangent digest mismatch (A42)"; exit 2; }
unset TT_MESH_GRAPH_DESC_PATH
source /home/ttuser/tt-bio-dev/env/bin/activate
ENV=(TT_BIO_SOFTMAX_BW_RENORM=1 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
     TT_BIO_LEASE_HOLDER=worker:bcx-heads OMP_NUM_THREADS=8 PYTHONPATH="$W")
CLK=$R/aiclk_${TAG}.txt; : > "$CLK"
( while true; do
    # sysfs node, not card index: on qb1 TT_VISIBLE_DEVICES 0,1,2,3 are nodes 1,2,3,0. tt-smi -s gave nothing.
    cat "/sys/class/tenstorrent/tenstorrent!${SL_AICLK_NODE:?sysfs node of this card}/tt_aiclk" >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
echo "=== $TAG start $(date -u +%FT%TZ) $(hostname) card $CARD commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain tt_bio | wc -l) load $(cut -d' ' -f1-3 /proc/loadavg)"
S=$(date +%s)
env "${ENV[@]}" timeout 5400 python3 perf/of3t_stackexact/stackarm.py --exact softmax,layer_norm \
    --stats-out "$R/STACK_EXACT_${TAG}.json" --census-out "$R/CENSUS_${TAG}.json" --lever none -- \
    --boundary "$B" --cap-last "$C" --out "$O/dev_${TAG}.pt" --report "$R/DEV_${TAG}.json" --arm flipped --crop 0 2>&1 \
  | tee "$O/raw_${TAG}.log" | grep --line-buffered -E '^\{|OUR GRADIENT|STACK_EXACT|Traceback|rror|FAILED' | cut -c1-400 | tail -25
rc=${PIPESTATUS[0]}
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $(( $(date +%s) - S ))s $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
python3 -c "import sys;s=sorted(int(l) for l in open('$CLK') if l.strip().isdigit());print('AICLK DURING n',len(s),'min',s[0],'median',s[len(s)//2],'max',s[-1]) if s else print('no clock')"
if [ -s "$O/dev_${TAG}.pt" ]; then
  sha256sum "$O/dev_${TAG}.pt" | tee "$R/dev_${TAG}.sha256"
  python3 "$BH/perf/bcx_heads/ptdigest.py" "$O/dev_${TAG}.pt" "$R/digest_${TAG}.json" && rm -f "$O/dev_${TAG}.pt"
fi
exit "$rc"
