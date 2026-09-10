#!/usr/bin/env bash
# boltzgen's p300c ceiling: all rungs on ONE card, ascending, stopping where it breaks.
#
# Round 3 left 1831 PASS and 3662 FAIL with a 1200.9 s wall equal to its own rung budget. Its log
# says the rung never ran slowly, it refused:
#   TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0)-(x=10,y=9)] grow to
#   1769984 B which is beyond max L1 size of 1572864 B
# raised as RuntimeError out of boltz.py:860 predict_step, from the triangle-multiplication matmul
# in the Pairformer, 310 s in -- then 19+ minutes unwinding tt-metal. A bigger budget buys a longer
# teardown, not a different answer, so the ladder now ends a rung at that line instead.
#
# 2745 halves the 1831..3662 interval: the circular-buffer limit is a step and 3662 is only 12.5 %
# over it, so the wall can sit anywhere between the two rungs.
#
# CARD: $1. Card 2 (this task's grant) is wedged -- "Device 0 init: failed to initialize FW" on
# every open, and the 256 rung that passed in 59.7 s now fails in 15 s, which is the negative
# control that says chip and not size. Clearing it needs tt-smi -r, which on qb2 resets the whole
# 2+3 board pair, and card 3 is mid-chain for capacity-size-ladder-reseed-p300c. So the walk fans
# onto the idle sibling with the grant widened to that card only.
set -u
CARD="${1:?card}"
cd /home/ttuser/.coworker/wt/p300c-boltzgen-ceiling
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
T=perf/bhdesign/targets/big_7324.cif
TT_BIO_LEASE_TIMEOUT=3600 BH_LEASE_CARDS="2,$CARD" $PY perf/bhdesign/ladder.py \
  --model boltzgen --sizes 1831,2745,3662,7324 --card "$CARD" --board p300c --target "$T" \
  --holder worker:p300c-boltzgen-ceiling --stop-on-fail --timeout 5400 \
  --out "$D/boltzgen.jsonl" --work "$D/work"
echo "WALK_BOLTZGEN_CEILING_DONE"
