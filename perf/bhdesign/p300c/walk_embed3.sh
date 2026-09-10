#!/usr/bin/env bash
# Round 3, card 2: the four embedding models that stopped at 65536 get a 98304 rung, so all six
# ceilings are bracketed by the same interval instead of four of them resting on a ladder top.
# 98304 x 98368 bf16 is 19.3 GB interleaved over 8 banks = 2.4 GB/bank, under the 3.65 GB block
# that 131072 (4.3 GB/bank) overruns -- so a failure here would have to come from WEIGHTS on top
# of the activation, which is the one place a 35M and a 6B model can legitimately differ.
set -u
cd /home/ttuser/.coworker/wt/p300c-1536-design-embed
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
for m in saprot-35m saprot-650m saprot-1.3b esmc-6b; do
  TT_BIO_LEASE_TIMEOUT=1800 BH_LEASE_CARDS=2 $PY perf/bhdesign/ladder.py \
    --model "$m" --sizes 98304 --card 2 --board p300c \
    --holder worker:p300c-1536-design-embed --timeout 1200 \
    --out "$D/$m.jsonl" --work "$D/work"
done
echo "WALK_EMBED3_DONE"
