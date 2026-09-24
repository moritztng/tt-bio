#!/bin/bash
# bcx-bwdplan chain 2 on qb1 card 0, after run_chain.sh, under one benchlock:
#   0. device kernel seconds of one Evoformer and one extra-MSA block backward, every arm in one
#      profiled process (run_prof.sh)
#   1. the mask arm: the whole 4+48 step pre-mask (`bwd`) and masked (`bwd+mask`) interleaved in
#      one process, from a tree that is this branch plus bcx-predictor's three masked sites
#      (4315acf94 and 3f457f9da, tt_bio/af2.py and perf/bcx_afgrad/afgrad.py only)
BP=/home/ttuser/.coworker/wt/bcx-bwdplan; MW=/dev/shm/bcx-bp-maskwt; R=$BP/perf/bcx_bwdplan
PY=/home/ttuser/tt-bio-dev/env/bin/python3
st(){ echo "== $1 $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) aiclk $(cat /sys/class/tenstorrent/*/tt_aiclk | tr '\n' ' ') df $(df -h / | awk 'NR==2{print $4}')"; }
cd $BP
st prof
bash perf/bcx_bwdplan/run_prof.sh --arms stack,bwd1,bwd2,bwd4,bwd --ks 1,2 --warm 1 --steps 2 2>&1 \
  | grep -E "^\{|^(stack|bwd)|EXIT|wrote|Error|Traceback|== "
cd $MW
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-bwdplan
export TMPDIR=/dev/shm/bcx-bp-tmp TT_METAL_CACHE=/dev/shm/bcx-bp-cache OMP_NUM_THREADS=8
O=/dev/shm/bcx-bp-out
echo "mask tree $(git rev-parse --short HEAD) + $(git diff --cached --stat | tail -1)"
for n in 256 224; do
  st "mask n$n"
  timeout 2400 taskset -c 16-31 $PY perf/bcx_stack/stack.py whole --arms bwd,bwd+mask --n $n --reps 3 \
    --seed 0 --out $O/mask_n${n}_s0.json 2>&1 | grep --line-buffered -E "^(bwd)|wrote|Error|Traceback"
  cp $O/mask_n${n}_s0.json $R/ 2>/dev/null
done
st done
