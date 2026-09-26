#!/bin/bash
cd /home/ttuser/.coworker/wt/of3t-p10wall || exit 1
mkdir -p perf/of3t_p10wall/out
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:of3t-p10wall
export PYTHONPATH=/home/ttuser/.coworker/wt/of3t-p10wall
{
  echo "=== launch $(date -u +%FT%TZ) ==="
  grep -E "MemTotal|MemAvailable" /proc/meminfo
  cat /proc/loadavg
} > perf/of3t_p10wall/out/wall_c4_8_12.log
/home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_stepfloor/fullstep.py \
  --tokens 384 --cycles 4 --samples 48 --reps 4 --chunk-per-rep 4,4,8,12 \
  --out perf/of3t_p10wall/out/wall_s48_cladder_384.json \
  >> perf/of3t_p10wall/out/wall_c4_8_12.log 2>&1
echo "EXIT=$?" >> perf/of3t_p10wall/out/wall_c4_8_12.log
