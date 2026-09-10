#!/usr/bin/env bash
# boltzgen's p300c ceiling. Round 3 left 3662 as a FAIL that its own 1200 s rung budget killed,
# so the number was a clock and not a wall: the run threw an L1 assert 143 s in, the engine
# absorbed it, and the process then held the card until the budget ran out. Same rung, 5400 s,
# and 7324 behind it if it passes. Card 2 only -- 1 and 3 are held by siblings.
set -u
cd /home/ttuser/.coworker/wt/p300c-boltzgen-ceiling
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
T=perf/bhdesign/targets/big_7324.cif
TT_BIO_LEASE_TIMEOUT=3600 BH_LEASE_CARDS=2 $PY perf/bhdesign/ladder.py \
  --model boltzgen --sizes 3662,7324 --card 2 --board p300c --target "$T" \
  --holder worker:p300c-boltzgen-ceiling --stop-on-fail --timeout 5400 \
  --out "$D/boltzgen.jsonl" --work "$D/work"
echo "WALK_BOLTZGEN_CEILING_DONE"
