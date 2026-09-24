#!/bin/bash
# Unprofiled walls and host attribution, card 0, one benchlock for the whole chain.
# qb1's root disk read 0 bytes free on 2026-09-24, so every write goes to tmpfs:
# outputs, TMPDIR (OpenMPI's session dir) and the JIT kernel cache.
cd /home/ttuser/.coworker/wt/bcx-realcensus
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-realcensus
export TMPDIR=/dev/shm/bcx-rc-tmp TT_METAL_CACHE=/dev/shm/bcx-rc-cache
O=/dev/shm/bcx-rc-out; mkdir -p $O $TMPDIR $TT_METAL_CACHE
PY=/home/ttuser/tt-bio-dev/env/bin/python3
R=perf/bcx_realcensus/realcensus.py
echo "== wheel wall $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
timeout 1500 $PY $R wall --stacks evo,extra --ks 1,2,4 --warm 2 --steps 6 --out $O/wall_wheel.json 2>&1 | grep -E "^\{|wrote|Error|Traceback"
echo "== wheel whole $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
timeout 2400 $PY perf/bcx_stack/stack.py whole --arms stack --n 256 --reps 3 --out $O/whole_n256_wheel.json 2>&1 | grep -E "^stack|wrote|Error|Traceback"
echo "== wheel host $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
timeout 1500 $PY $R host --stacks evo,extra --ks 1 --warm 2 --out $O/host_wheel.json 2>&1 | grep -E "^\{|wrote|Error|Traceback"
echo "== mar31 wall $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
M=/home/ttuser/tt-metal-main-mar31
( export TT_METAL_HOME=$M PYTHONPATH=$M/ttnn:$M/tools:$M; unset LD_LIBRARY_PATH
  timeout 1500 $PY $R wall --stacks evo,extra --ks 1,2 --warm 2 --steps 6 --out $O/wall_mar31.json 2>&1 | grep -E "^\{|wrote|Error|Traceback" )
echo "== done $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
