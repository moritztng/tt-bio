#!/bin/bash
set -u
WT=/home/ttuser/.coworker/wt/sizes-recheck-boltz2-esmfold2
cd "$WT" || exit 1
trap "cd $WT && git checkout -q wk/sizes-recheck-boltz2-esmfold2 2>&1 | tail -2" EXIT
PY=/home/ttuser/tt-bio-dev/env/bin/python3
run () { # $1=tag
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:sizes-recheck-boltz2-esmfold2 PYTHONPATH=$WT \
    $PY perf/other512/fold_ab_multi.py --model boltz2 --sizes 128 --arms on,on,on \
    --out /tmp/endpoint_$1.json 2>&1 | grep -E "^  on:|^=== |K1 |K2 |E6 |wrote"
}
echo "##### LEG A: 2026-08-13 tip a3a5a94f"
git checkout -q --detach a3a5a94f || { echo CHECKOUT_FAIL; exit 1; }
git log --oneline -1
run old
echo "##### LEG B: today main (branch tip)"
git checkout -q wk/sizes-recheck-boltz2-esmfold2
git log --oneline -1
run new
