#!/bin/bash
# Session 2: the 768 and 1024 aa curve points, one arm per process.
#
# Why one arm per process. Above ~640 aa the second fold in a process stalls the device in the MSA
# stack (state doc 7.1: py-spy shows MainThread blocked in ttnn.synchronize_device, 78 min at 1024
# and 11.5 min at 768, zero TT_THROWs, same configuration as the fold that completed). So the only
# fold that returns at these sizes is the FIRST one. --skip-cold makes arm 1 that fold, with the
# full counter record attached.
#
# What that costs. The first fold in a process is cold, and cold carries a one-time additive cost
# MEASURED at 45.9 s at 128 aa (53.92 cold vs 8.04 warm) and 38.1 s at 256 aa (55.81 vs 17.68). An
# additive constant on both arms COMPRESSES the ratio toward 1.0, so every ratio this session
# produces is a LOWER BOUND on the warm ratio the 512 and 640 points are quoted at. It is not
# directly comparable to them and the state doc must say so wherever it is quoted.
#
# REGISTERED PREDICTIONS (written before the run; do not edit after it starts):
#   P1  768 base4 cold fold: 208-222 s, i.e. a cold ratio of 1.04-1.11 against the already
#       MEASURED post-fix 768 `on` cold fold of 199.63 s.
#   P2  768 `on` cold fold re-run: 199.63 s +/- 4 s. This is the only A/A available at this size
#       and it is a CROSS-PROCESS one. If it misses by more than 4 s, the cold-fold comparison is
#       not a valid instrument and P1/P3/P4 must all be discarded rather than quoted.
#   P3  1024 `on` cold fold: 300-330 s. The old 315.09 s reading is NOT the shipped stack -- it was
#       taken at 01:16, four minutes before the E6 fix ecbbb78e, so its `on` arm ran E6 disabled.
#       This run is the first correct `on` at 1024. Expect it at or slightly below 315.09.
#   P4  1024 base4 cold fold: 315-345 s, i.e. a cold ratio of 1.00-1.08.
#   P5  The curve continues monotone decreasing in N: cold ratio at 1024 < cold ratio at 768.
#   Stop rule: any ratio inside the P2 A/A spread is reported as "not resolved", not as a number.
#
# Order is highest-value-first: 768 base4 lands the 768 ratio on its own, because that size's `on`
# cold fold is already measured post-fix.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-sizes-perf
cd $WT
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:protenix-v2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=perf/pxsizes

sha256sum $WT/perf/size512/fixtures/cdk2x2_{768,1024}.{yaml,a3m}

run () {  # run <size> <arm> <out>
  echo "=== $(date -Is)  size $1  arm $2  (one arm, cold) ==="
  timeout 3000 $PY -u perf/size512/fold_ab512.py --sizes "$1" --arms "$2" --skip-cold \
      --timers full --out "$3"
  echo "RC_$1_$2=$?"
}

run 768  base4 $O/s2_768_base4.json
run 768  on    $O/s2_768_on.json
run 1024 on    $O/s3_1024_on.json
run 1024 base4 $O/s3_1024_base4.json
echo "=== $(date -Is) session 2 done ==="
