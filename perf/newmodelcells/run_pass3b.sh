#!/bin/bash
# Re-run the arm the killed ssh took with it, plus a third OpenFold3 control taken in the quiet
# window, so the of3_1 arm measured under load is bracketed rather than argued about.
set -u
OWNER=worker:perf-page-newmodels-fold-cells
WT=/home/ttuser/.coworker/wt/perf-page-newmodels-fold-cells
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=3
OUT=$WT/perf/newmodelcells
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=$OWNER
export PATH=$HOME/.local/bin:$PATH
cd "$WT" || exit 1
for pair in ob_2:openbind of3_3:openfold3; do
  lab=${pair%%:*}; mdl=${pair##*:}
  echo "=== $lab ($mdl) $(date -u +%FT%TZ) load=$(cut -d" " -f1-3 /proc/loadavg) ==="
  $PY "$WT/perf/of3_4xpd/xmodel_ab.py" --model "$mdl" --tree "$WT" --size 512 --repeat 3 \
      --label "$lab" --out "$OUT/ob_${mdl}_${lab}.json" || echo "$lab rc=$?"
done
echo "=== done $(date -u +%FT%TZ) load=$(cut -d" " -f1-3 /proc/loadavg) ==="
