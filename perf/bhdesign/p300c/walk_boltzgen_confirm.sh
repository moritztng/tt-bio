#!/usr/bin/env bash
# Leg two: the ceiling's PASS side, on the same card as its FAIL side.
#
# Round 3 proved 1831 on card 3 and refused 3662 there. A ceiling whose two legs sit on different
# chips is two measurements, not one, so 1831 is re-run here on card 2 before 3662's refusal is
# called a wall. 2745 follows it because the L1 circular-buffer limit is a step, not a gradual
# fill: 3662 asks 1769984 B against 1572864 B, only 12.5 % over, so the wall could sit anywhere in
# 1831..3662 and a midpoint halves the interval the release gate has to land.
set -u
cd /home/ttuser/.coworker/wt/p300c-boltzgen-ceiling
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
T=perf/bhdesign/targets/big_7324.cif
# Wait out the 3662 rung: one device context per process, and its teardown still holds card 2.
while kill -0 "${1:?walk pid}" 2>/dev/null; do sleep 20; done
TT_BIO_LEASE_TIMEOUT=3600 BH_LEASE_CARDS=2 $PY perf/bhdesign/ladder.py \
  --model boltzgen --sizes 1831,2745 --card 2 --board p300c --target "$T" \
  --holder worker:p300c-boltzgen-ceiling --stop-on-fail --timeout 5400 \
  --out "$D/boltzgen.jsonl" --work "$D/work"
echo "WALK_BOLTZGEN_CONFIRM_DONE"
