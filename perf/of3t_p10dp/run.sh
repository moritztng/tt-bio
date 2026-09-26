#!/bin/bash
# of3t-p10dp: one timed arm. Rooted in THIS row's worktree so fleet hygiene cannot delete the
# cwd out from under a detached run.
W=/home/ttuser/.coworker/wt/of3t-p10dp
O=$W/perf/of3t_p10dp/out
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W" || exit 1
mkdir -p "$O"

TAG=${1:?usage: run.sh TAG CARD python-args...}; shift
CARD=${1:?usage: run.sh TAG CARD python-args...}; shift

# The clock, sampled DURING and off the class node, independently of the harness's own
# sampler. A perf number on Blackhole without the clock it was measured at is not a
# measurement: the AICLK sets the fold time.
CLK=$O/aiclk_${TAG}.txt
: > "$CLK"
( while true; do cat "/sys/class/tenstorrent/tenstorrent!$CARD/tt_aiclk" >> "$CLK" 2>/dev/null; sleep 1; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

unset TT_MESH_GRAPH_DESC_PATH
echo "=== $TAG start $(date -u +%FT%TZ) $(hostname) card $CARD loadavg $(cut -d' ' -f1-3 /proc/loadavg) ==="
env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-p10dp \
    TT_BIO_SOFTMAX_BW_FP32=1 OMP_NUM_THREADS=8 PYTHONPATH="$W" \
    timeout 5400 "$PY" "$@"
rc=$?
echo "=== $TAG done $(date -u +%FT%TZ) rc=$rc ==="
kill $SAMPLER 2>/dev/null || true
# min/max/median of the independent sampler, so the clock claim does not rest on one instrument
sort -n "$CLK" | awk '{v[NR]=$1} END {if(NR)printf "AICLK %s: n=%d min=%d median=%d max=%d\n","'"$TAG"'",NR,v[1],v[int((NR+1)/2)],v[NR]}'
exit $rc
