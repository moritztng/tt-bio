#!/usr/bin/env bash
# boltzgen's p300c ceiling, all rungs on card 2 in one ascending walk that stops where it breaks.
#
# Round 3 left 1831 PASS on card 3 and 3662 FAIL on card 3 with a 1200 s wall, which was its own
# rung budget. Re-reading that rung's log shows it was never a slow run: it threw
#   TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0)-(x=10,y=9)] grow to
#   1769984 B which is beyond max L1 size of 1572864 B
# 310 s in, raised RuntimeError out of boltz.py:860 predict_step, and then spent 19+ minutes
# unwinding tt-metal. So a bigger budget buys a longer teardown, not a different answer; the
# ladder now ends a rung at that terminal line instead. 2745 halves the 1831..3662 interval,
# because the circular-buffer limit is a step and 3662 is only 12.5 % over it.
set -u
cd /home/ttuser/.coworker/wt/p300c-boltzgen-ceiling
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
T=perf/bhdesign/targets/big_7324.cif
TT_BIO_LEASE_TIMEOUT=3600 BH_LEASE_CARDS=2 $PY perf/bhdesign/ladder.py \
  --model boltzgen --sizes 1831,2745,3662,7324 --card 2 --board p300c --target "$T" \
  --holder worker:p300c-boltzgen-ceiling --stop-on-fail --timeout 5400 \
  --out "$D/boltzgen.jsonl" --work "$D/work"
echo "WALK_BOLTZGEN_CEILING_DONE"
