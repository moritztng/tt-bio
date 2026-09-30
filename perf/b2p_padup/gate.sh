#!/bin/bash
# Release gate on wk/b2p-padup, qb2 card 0 (p300c), after bc2p300c.sh. The pad-up is on by default in this tree.
cd /home/ttuser/.coworker/wt/b2p-padup
until grep -q "^=== DONE" perf/b2p_padup/out/bc2/ladder.log 2>/dev/null; do sleep 15; done
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2p-padup
git rev-parse HEAD > perf/b2p_padup/out/gate.sha
timeout 5400 ~/tt-bio-dev/env/bin/python3 scripts/release_gate.py --model openfold3 --model boltz2 \
  --model size-ladder --size-ladder-models openfold3,boltz2 --size-ladder-rungs 544,608 \
  --load-ceiling 64 --keep > perf/b2p_padup/out/gate.log 2>&1
echo "gate rc=$?" >> perf/b2p_padup/out/gate.log
