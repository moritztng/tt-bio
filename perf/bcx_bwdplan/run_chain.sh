#!/bin/bash
# bcx-bwdplan timing chain on qb1 card 0, one benchlock for the whole chain. qb1's root disk is
# ~1 G free, so TMPDIR, the JIT cache and outputs sit on tmpfs; the small JSONs are copied back.
#   0. one block per stack, every arm against the base arm: bit-identical or rel L2
#   1. per block (Evoformer, extra-MSA), checkpointed as the step runs it, K=1,2, arms interleaved
#   2. the whole 4+48 step at n=224 and n=256, base (`stack`) and lever (`bwd`) interleaved, seed 0
cd /home/ttuser/.coworker/wt/bcx-bwdplan
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-bwdplan
export TMPDIR=/dev/shm/bcx-bp-tmp TT_METAL_CACHE=/dev/shm/bcx-bp-cache OMP_NUM_THREADS=8
O=/dev/shm/bcx-bp-out; R=perf/bcx_bwdplan; mkdir -p $O $TMPDIR $TT_METAL_CACHE
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PIN="taskset -c 16-31"
S=perf/bcx_stack/stack.py
st(){ echo "== $1 $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg) aiclk $(cat /sys/class/tenstorrent/*/tt_aiclk | tr '\n' ' ')"; }
st bits
timeout 1200 $PIN $PY $S bits --n 256 --arms stack,bwd1,bwd2,bwd4,bwd --out $O/bits_n256.json 2>&1 | grep -E "wrote|Error|Traceback"
cp $O/bits_n256.json $R/ 2>/dev/null
st time
timeout 1500 $PIN $PY $S time --ckpt --ns 256 --stacks evo,extra --ks 1,2 --warm 2 --steps 6 \
  --arms stack,bwd1,bwd2,bwd4,bwd --out $O/time_n256.json 2>&1 | grep -E "^\{|wrote|Error|Traceback"
cp $O/time_n256.json $R/ 2>/dev/null
for n in 224 256; do
  st "whole n$n"
  timeout 2400 $PIN $PY $S whole --arms stack,bwd --n $n --reps 3 --seed 0 --out $O/whole_n${n}_s0.json 2>&1 \
    | grep -E "^(stack|bwd)|wrote|Error|Traceback"
  cp $O/whole_n${n}_s0.json $R/ 2>/dev/null
done
st done
