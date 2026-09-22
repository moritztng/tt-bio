#!/bin/bash
# of3t-stepfloor pass 3: the lever priced at the op, once arm B is off the card.
cd /home/ttuser/.coworker/wt/of3t-stepfloor || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/stepfloor
O=perf/of3t_stepfloor/out
while pgrep -f "of3t_stepfloor/fullstep.py" >/dev/null; do sleep 15; done
echo "=== renorm_micro start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> "$L/arms3.log"
env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-stepfloor \
    $PY perf/of3t_stepfloor/renorm_micro.py --tokens 384 --iters 40 --blocks 4 \
    --out $O/renorm_micro_384.json > "$L/renorm_micro.log" 2>&1
echo "=== renorm_micro done  $(date -u +%FT%TZ) rc=$?" >> "$L/arms3.log"
