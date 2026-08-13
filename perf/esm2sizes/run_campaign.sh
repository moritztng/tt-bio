#!/usr/bin/env bash
# One benchlock hold, three decompositions: 1024 (capacity + baseline), 128 (census, cheap
# inertness check), 768 (the untested edge of two windows). Sequential with `;` not `&&` so a
# 1024 capacity refusal does not cancel the rest -- a refusal IS the finding for that size.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-sizes-perf
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
ENVP="env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:esmfold2-sizes-perf PYTHONPATH=$WT"
D=perf/esm512/decomp.py
O=perf/esm2sizes

echo "=== campaign start $(date -Is) ==="
echo "--- 1024 aa: capacity probe + baseline ---"
$ENVP $PY $D --size 1024 --plain 1 --timed 1 --out $O/decomp_1024_c2.json 2>&1
echo "RC_1024=$?"
echo "--- 128 aa: census (re-run after the record() fix) ---"
$ENVP $PY $D --size 128 --plain 2 --timed 1 --out $O/decomp_128b_c2.json 2>&1
echo "RC_128=$?"
echo "--- 768 aa: both windows at their edge ---"
$ENVP $PY $D --size 768 --plain 1 --timed 1 --out $O/decomp_768_c2.json 2>&1
echo "RC_768=$?"
echo "=== campaign end $(date -Is) ==="
