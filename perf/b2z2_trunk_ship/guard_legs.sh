#!/bin/bash
# The guard only changes what happens when a pair-tensor axis is one tile, i.e. chains of at most
# 32 residues. These are the remaining gate legs whose target is that short; boltz2-trpcage-nomsa
# already re-ran and reproduces main's digits.
set -u
WT=/home/ttuser/.coworker/wt/b2z2-trunk-byte-round2-ship
cd "$WT" || exit 1
for leg in esmfold2-trpcage esmfold2-fast-trpcage opendde-trpcage-nomsa; do
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
  TT_BIO_LEASE_HOLDER=worker:b2z2-trunk-byte-round2-ship PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/full_parity_gate.py --leg "$leg" --fresh \
    --workers localhost:0 --load-ceiling 999 --workdir perf/b2z2_gate/guard_short_work \
    --out "perf/b2z2_gate/gate_guard_$leg.json" > "perf/b2z2_gate/guard_$leg.log" 2>&1
  echo "== $leg rc=$? $(python3 -c "import json;d=json.load(open('perf/b2z2_gate/gate_guard_$leg.json'));print([(l['leg'],l['verdict'],l.get('detail')) for l in d['legs']])" 2>/dev/null)"
done
echo SHORTLEGSDONE
