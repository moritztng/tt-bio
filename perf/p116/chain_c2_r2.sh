#!/bin/bash
# card 2: R2 (3844 atoms) ladder, then the two acceptance checks at b=6 -- the batch a cap
# deletion would actually deliver at this size, since the atom-pair budget admits 6 there.
# Card 2 was verified idle (no fuser holder) before launch; the grant is widened for this chain
# only, and both TT_VISIBLE_DEVICES and TT_BIO_LEASE_CARDS name it so an unpinned open still
# fails rather than bringing up the whole box.
cd /home/ttuser/.coworker/wt/rfd3-b8-to-4x-p4 || exit 1
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_CARDS=1,2
export TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p4
PY=/home/ttuser/.coworker/rel070/relvenv/bin/python3
L=perf/p116/c3_card2.log
echo "=== card2 chain start $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg) ===" > $L
$PY -u scripts/rfd3_port/p116_cap_ladder.py \
  --spec perf/dsfix/fixtures/rfd3_R2.json --out perf/p116/ladder_R2_c3.json \
  --reps 3 --arms 1,6 --warm-steps 25 >> perf/p116/R2_c3.log 2>&1
echo "=== ladder R2 exit $? $(date -u +%H:%M:%S) ===" >> $L
$PY -u scripts/rfd3_port/verify_batch_trajectory_parity.py \
  --spec perf/dsfix/fixtures/rfd3_R2.json --batch 6 --timesteps 8 \
  >> perf/p116/parity_R2_b6.log 2>&1
echo "=== parity R2 b6 exit $? $(date -u +%H:%M:%S) ===" >> $L
$PY -u scripts/rfd3_port/p25_dram_headroom.py \
  --spec perf/dsfix/fixtures/rfd3_R2.json --batches 1 6 --timesteps 3 \
  --json perf/p116/dram_R2.json >> perf/p116/dram_R2.log 2>&1
echo "=== dram R2 exit $? $(date -u +%H:%M:%S) ===" >> $L
