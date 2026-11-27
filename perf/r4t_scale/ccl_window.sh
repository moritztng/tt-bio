#!/bin/bash
# r4t-scale D89: whole-box window on anx-glx. Waits for 19:00Z and zero /dev/tenstorrent holders
# (gives up 19:20Z touching nothing), glx_reset (owed for wedged 29,30), then the device all-reduce
# bench on the 8x4 Ring fabric at submesh widths 4/8/16/32. Stops by 19:45Z. Log ccl/window.log.
cd ~/fleet/r4t/ccl; L=window.log; log(){ echo "$(date -u +%T) $*" >> $L; }
holders(){ for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q tenstorrent && echo ${p#/proc/}; done; }
until [ $(date -u +%H%M) -ge 1900 ] && [ -z "$(holders)" ]; do
  [ $(date -u +%H%M) -ge 1920 ] && { log "GAVE UP: holders $(holders)"; exit 3; }
  [ -e ~/fleet/HANDBACK ] && { log HANDBACK; exit 4; }; sleep 30; done
log "box free, glx_reset"; timeout 900 tt-smi -glx_reset > reset.log 2>&1; log "glx_reset rc=$?"
PY=~/fleet/env/bin/python
run(){ # n sub arms
  [ $(date -u +%H%M) -ge 1940 ] && { log "skip n=$1 (window end)"; return; }
  FABRIC=FABRIC_1D_RING SUB=$2 timeout 600 $PY ccl_allreduce.py $1 8x4 ring$1.json $3 3 > ring$1.log 2>&1
  log "ring n=$1 sub=$2 rc=$? $(grep -E '^(a2a|a2abroken|builtin|agadd|broken) ' ring$1.log | awk '{print $1}' | tr '\n' ' ')"; }
run 4 1x4 a2a,a2abroken,builtin
run 32 8x4 a2a,a2abroken,builtin,agadd
run 8 2x4 a2a,a2abroken,builtin
run 16 4x4 a2a,a2abroken,builtin
log "window end"
