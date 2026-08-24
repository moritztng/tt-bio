#!/bin/bash
# One benchlock hold, two measurements: the Job-1 gate first (short), then the baseline
# verification (long). Chained so a single acquisition on a contended box buys both.
set -u
cd /home/ttuser/.coworker/wt/rfd3-b8-to-4x-p3
E="env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p3 PYTHONPATH=/home/ttuser/.coworker/wt/rfd3-b8-to-4x-p3"
PY=/home/ttuser/tt-bio-dev/env/bin/python3
echo "=== [1/2] p79 gathered probe  $(date -Is) ==="
$E $PY -u scripts/rfd3_port/p79_gathered_probe.py perf/p79/gathered_probe.json 6051 128 4 5
echo "rc=$? ($(date -Is))"
echo "=== [2/2] p68 baseline verify  $(date -Is) ==="
$E $PY -u scripts/rfd3_port/p68_stack_ab.py perf/p79/baseline_verify.json 200 off,off,on,on,on
echo "rc=$? ($(date -Is))"
