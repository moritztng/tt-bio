#!/bin/bash
# Session 1 of the protenix-v2 size sweep: Phase 0 + the decomposition at 128 and 256, then the
# 1024 aa capacity answer. One benchlock hold, four processes, one device context each.
#
# Order is cheap-first on purpose. 128 and 256 fold in seconds, so they land the lever table and
# both screens even if the turn ends early; 1024 is the priority answer but costs ~5 min a fold.
# Every process writes its JSON after every fold, so a truncated session still banks what it ran.
#
# Registered before the run (state/protenix-v2-sizes-perf.md S1/S3/S4):
#   S1 bigchunk64 at 128:  -0.08 to -0.20 s/fold, all of it in body:Transition
#   S3 l1max224  at 256:   a LOSS of 0.1 to 0.4 s/fold
#   S4 base4/on:           below 1.10x at 128 and 256, 1.10-1.20x at 768 and 1024
#   Phase 0: E6 and E2 gated off at 128 and 256 (L1_N_MIN=288 > N), K1/K1b/K2 fire everywhere,
#            wide-q inert at 128 and 256, transpose lever dead at all four.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-sizes-perf
cd $WT
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:protenix-v2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=perf/pxsizes

sha256sum $WT/perf/size512/fixtures/cdk2x2_{128,256,768,1024}.{yaml,a3m}

run () {  # run <size> <arms> <out>
  echo "=== $(date -Is)  size $1  arms $2 ==="
  $PY -u perf/size512/fold_ab512.py --sizes "$1" --arms "$2" --timers full --out "$3"
  echo "RC_$1=$?"
}

run 128  on,on,base4,noe6,nohmqkv,nok2,narrowq,off,bigchunk64,on $O/s1_128.json
run 256  on,on,base4,noe6,nohmqkv,nok2,narrowq,off,l1max224,on   $O/s1_256.json
run 1024 on,off                                                  $O/s3_1024.json
run 768  on,on,base4,nok2,off                                    $O/s2_768.json
echo "=== $(date -Is) session 1 done ==="
