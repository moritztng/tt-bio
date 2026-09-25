#!/bin/bash
# bcx-bwdplan chain 3 on qb1 logical 0 (node 1, 01:00.0), one benchlock:
#   1. the mask arm, `whole --arms bwd,bwd+mask` at n=256 then n=224, from a tree that is this
#      branch plus bcx-predictor 4315acf94, 3f457f9da and 25564fee2 (the OPM without ttnn.repeat)
#   2. device kernel seconds per block backward, arms stack,bwd1,bwd4,bwd, K=1, one Tracy process.
#      Chain 2 profiled five arms at K=1,2 and never got its device report; this is realcensus's
#      op count.
# Heartbeat and AICLK of the card are stamped at both ends of every window.
BP=/home/ttuser/.coworker/wt/bcx-bwdplan; MW=/dev/shm/bcx-bp-maskwt; R=$BP/perf/bcx_bwdplan
PY=/home/ttuser/tt-bio-dev/env/bin/python3; N='/sys/class/tenstorrent/tenstorrent!1'
st(){ echo "== $1 $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) aiclk $(cat "$N/tt_aiclk") hb $(cat "$N/tt_heartbeat") df $(df -h / | awk 'NR==2{print $4}')"; }
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-bwdplan
export TMPDIR=/dev/shm/bcx-bp-tmp TT_METAL_CACHE=/dev/shm/bcx-bp-cache OMP_NUM_THREADS=8
O=/dev/shm/bcx-bp-out
cd $MW || exit 2
echo "mask tree $(git rev-parse --short HEAD) + $(git diff --cached --stat | tail -1)"
for n in 256 224; do
  st "mask n$n"
  timeout 3600 taskset -c 16-31 $PY perf/bcx_stack/stack.py whole --arms bwd,bwd+mask --n $n --reps 3 \
    --seed 0 --out $O/mask_n${n}_s0.json 2>&1 | grep --line-buffered -E "^(bwd)|wrote|Error|Traceback"
  cp $O/mask_n${n}_s0.json $R/ 2>/dev/null
  st "mask n$n end"
done
cd $BP || exit 2
st prof
bash perf/bcx_bwdplan/run_prof.sh --arms stack,bwd1,bwd4,bwd --ks 1 --warm 1 --steps 2 2>&1 \
  | grep -E "^(stack|bwd)|EXIT|wrote|Error|Traceback|== "
st done
