#!/bin/bash
# spd-diffusion on .107 (held WH Galaxy), chips 10/11, from ~/spd/spd-diffusion (eng/int, eng/diff, spd-data).
# Chip 10: conditioning once, then base vs lever (exact), then lever traced. Chip 11 waits for the
# conditioning, then the same under TT_BIO_LPX=1. Usage: wh107.sh 10|11 [RUNNAME]
. ~/japanfold/env.sh
B=~/spd/spd-diffusion; R=$B/runs/${2:-wh1}; mkdir -p $R ~/spd/locks
export SPD_DATA=$B/spd-data TT_BIO_LEASE_DIR=$B/leases TT_BIO_LEASE_HOLDER=spd-diffusion
RUN=$B/eng/diff/perf/spd_diffusion/run.sh
L=~/spd/locks/chip$1.lock
case $1 in
10)
  ( export TT_VISIBLE_DEVICES=10 PYTHONPATH=$B/eng/diff
    flock -w 600 $L timeout 2400 python $B/eng/diff/perf/spd_diffusion/bench.py cond $R 10 $SPD_DATA/inputs/c730.yaml > $R/cond.log 2>&1 )
  echo "$(date -u +%FT%TZ) cond rc=$?" >> $R/driver.log
  flock -w 600 $L $RUN $R/c10 10 $R/cond.pt base=$B/eng/int lever=$B/eng/diff
  SPD_TRACE=1 flock -w 600 $L $RUN $R/c10t 10 $R/cond.pt lever_trace=$B/eng/diff ;;
11)
  for i in $(seq 1 120); do [ -s $R/cond.pt ] && break; sleep 30; done
  flock -w 600 $L $RUN $R/c11 11 $R/cond.pt base_lpx=$B/eng/int:TT_BIO_LPX=1 lever_lpx=$B/eng/diff:TT_BIO_LPX=1
  SPD_TRACE=1 flock -w 600 $L $RUN $R/c11t 11 $R/cond.pt lever_lpx_trace=$B/eng/diff:TT_BIO_LPX=1 ;;
esac
echo "$(date -u +%FT%TZ) chip $1 end" >> $R/driver.log
