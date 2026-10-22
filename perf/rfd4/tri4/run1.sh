#!/bin/bash
# tri4 run1: T1 levers standalone at 512 (tri2/bench.py), RFD4's default set as the base, no mask (the trunk
# passes none when every token is real), each arm twice in A..B..A..B order.
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-tri4; O=$W/artifacts/tri4/run1; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri4 PYTHONPATH=$W
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)"; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 10800 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git log --oneline -1 | cut -c1-70)"
N=trimul_t1+pair_mm+pair_mm_tr+transition_b8+trimul_gin+trimul_gin_w8+pair_block+pair_tail+trimul_t1_dram+pair_bias_rms+pair_write_res
P1="set:trimul_tail.T1_PLANS_W8={(12,16,12):(1,8)}"
P2="set:trimul_tail.T1_PLANS_W8={(12,16,12):(1,16)}"
ARMS="lever:$N;mask:none|lever:$N+trimul_t1_psig;mask:none|lever:$N+trimul_gin_hifi2;mask:none"
ARMS="$ARMS|lever:$N+trimul_t1w8;mask:none|lever:$N+trimul_t1w8;$P1;mask:none|lever:$N+trimul_t1w8;$P2;mask:none"
ARMS="$ARMS|lever:$N+trimul_t1_psig+trimul_gin_hifi2+trimul_t1w8;$P1;mask:none"
timeout 3000 $PY perf/rfd4/tri2/bench.py --weights $HOME/rfd4-bundle --n 512 --reps 30 --out $O/bench.json \
    --arms "$ARMS|$ARMS" > $O/bench.log 2>&1
stamp "bench rc=$?"
stamp end
