#!/bin/bash
# One benchlock hold, both page cells. PXDesign first: it is the cheaper run and the row that
# matters, so its JSONs land early even if the hold is cut short.
set -u
OWNER=worker:perf-page-newmodels-fold-cells
WT=/home/ttuser/.coworker/wt/perf-page-newmodels-fold-cells
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=3
OUT=$WT/perf/newmodelcells
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=$OWNER
export PATH=$HOME/.local/bin:$PATH
cd "$WT" || exit 1
echo "=== driver start $(date -u +%FT%TZ) load=$(cut -d" " -f1-3 /proc/loadavg) ==="
for leg in A B; do
  echo "=== PXDESIGN leg $leg $(date -u +%FT%TZ) load=$(cut -d" " -f1-3 /proc/loadavg) ==="
  $PY "$OUT/pxd_pagecell.py" --tree "$WT" --yaml "$OUT/laczc_512_tt.yaml" \
      --n_step 400 --rounds 5 --label "px_$leg" --out "$OUT/px_$leg.json" || echo "PX $leg rc=$?"
done
for pair in of3_1:openfold3 ob_1:openbind of3_2:openfold3 ob_2:openbind; do
  lab=${pair%%:*}; mdl=${pair##*:}
  echo "=== $lab ($mdl) $(date -u +%FT%TZ) load=$(cut -d" " -f1-3 /proc/loadavg) ==="
  $PY "$WT/perf/of3_4xpd/xmodel_ab.py" --model "$mdl" --tree "$WT" --size 512 --repeat 3 \
      --label "$lab" --out "$OUT/ob_${mdl}_${lab}.json" || echo "$lab rc=$?"
done
echo "=== driver end $(date -u +%FT%TZ) load=$(cut -d" " -f1-3 /proc/loadavg) ==="
