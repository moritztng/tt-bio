#!/bin/sh
cd /home/ttuser/.coworker/wt/pairformer-resident-chunking || exit 1
uptime
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:pairformer-resident-chunking
echo "=== ARM --fast ==="
~/tt-bio/env/bin/python3 perf/bigswing/trunk_trace_probe.py --n 512 --fast \
  --out perf/bigswing/trace_probe_512_fast_qb2c0.json
echo "rc_fast=$?"
uptime
echo "=== ARM default ==="
~/tt-bio/env/bin/python3 perf/bigswing/trunk_trace_probe.py --n 512 \
  --out perf/bigswing/trace_probe_512_qb2c0.json
echo "rc_def=$?"
uptime
echo "=== ALLDONE ==="
