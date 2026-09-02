#!/bin/bash
# The last rung still measured under load: b=32.
#
# The per-round loadavg in the ladder JSONs says the regime split in the first control chain was
# read wrong. c400_n64 warm rounds ran at loadavg 1.00 and 1.08, so b=64 is already quiet, and
# d400_n8r5 rounds 2 and 3 ran at 8.50 and 9.84, so the b=8 anchor was not. c400_n32 is loaded
# throughout: 7.09 and 8.77 on its two warm rounds. It is the only rung left whose published number
# comes from the loaded regime, and the measured load penalty on this box is batch-dependent
# (9.6 % at b=2, 5.8 % at b=4, 2.6 % at b=16), so it cannot be corrected for, only re-measured.
#
# Cold plus two warm, the same shape c400_n32 used. About thirty-two minutes.
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-batchcurve-qb2-reverify
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OWNER=worker:pxdesign-batchcurve-qb2-reverify
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=$OWNER PYTHONPATH=$WT

out=perf/newmodelcells/qb2_batchcurve400/ctl400_n32.json
if [ -f "$out" ] && $PY -c "import json,sys; sys.exit(0 if json.load(open(\"$out\")).get(\"warm_n\") else 1)" 2>/dev/null; then
  echo "=== ctl400_n32 already measured, skipped"
else
  echo "=== ctl400_n32 start $(date -u +%FT%TZ) load=$(cut -d\" \" -f1-3 /proc/loadavg)"
  $PY perf/newmodelcells/pxd_pagecell.py --tree "$WT" \
      --yaml perf/newmodelcells/laczc_512_tt.yaml --n_step 400 --n_sample 32 --rounds 3 \
      --label ctl400_n32 --out "$out"
  echo "=== ctl400_n32 rc=$? $(date -u +%FT%TZ)"
fi
echo "QB2_CTL32_DONE $(date -u +%FT%TZ)"
