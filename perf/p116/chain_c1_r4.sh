#!/bin/bash
# card 1: R4 (6051 atoms) ladder, then the two acceptance checks at the only batched arm the
# atom-pair budget admits there (b=2). Rooted in this worker's own worktree.
cd /home/ttuser/.coworker/wt/rfd3-b8-to-4x-p4 || exit 1
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p4
PY=/home/ttuser/.coworker/rel070/relvenv/bin/python3
L=perf/p116/c3_card1.log
echo "=== card1 chain start $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg) ===" > $L
$PY -u scripts/rfd3_port/p116_cap_ladder.py \
  --spec perf/dsfix/fixtures/rfd3_R4.json --out perf/p116/ladder_R4_c3.json \
  --reps 3 --arms 1,2 --warm-steps 25 >> perf/p116/R4_c3.log 2>&1
echo "=== ladder R4 exit $? $(date -u +%H:%M:%S) ===" >> $L
$PY -u scripts/rfd3_port/verify_batch_trajectory_parity.py \
  --spec perf/dsfix/fixtures/rfd3_R4.json --batch 2 --timesteps 8 \
  >> perf/p116/parity_R4_b2.log 2>&1
echo "=== parity R4 b2 exit $? $(date -u +%H:%M:%S) ===" >> $L
$PY -u scripts/rfd3_port/p25_dram_headroom.py \
  --spec perf/dsfix/fixtures/rfd3_R4.json --batches 1 2 --timesteps 3 \
  --json perf/p116/dram_R4.json >> perf/p116/dram_R4.log 2>&1
echo "=== dram R4 exit $? $(date -u +%H:%M:%S) ===" >> $L
