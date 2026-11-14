#!/bin/bash
# census.sh OUT CARD TREE [FAST_LEVERS] (on qb1): OpenDDE cdk2x2 512 op census on one p150a, normal then fast.
# Per mode: a Tracy device profile of a 1-recycle, 2-step fold (perf/mol1/bh/opgap.py prof; the per-block and
# per-step ops are those of the full fold) and the served fold with each trunk block synced (opgap.py wall).
# FAST_LEVERS (comma list) replaces the fast arm's lever set; normal runs the tree's own BH set.
# Books the card through ~/spd/feed/book.py under its own owner, holds the card flock, drops the booking on exit.
set -u
OUT=$1 CHIP=$2 T=$3 FL=${4:-}
M=$HOME/tt-metal-main-mar31 PY=$HOME/tt-bio-dev/env/bin/python3 BK=$HOME/spd/feed/book.py
OWN=spd-m-opendde-2-c$CHIP-census FX=$T/perf/size512/fixtures
mkdir -p "$OUT"; log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/chain.log"; }
trap 'python3 $BK rm $OWN; log "chain exit"' EXIT
log "book card $CHIP: $(python3 $BK add $OWN $CHIP 3600 --note "OpenDDE BH op census 512" | tail -1)"
while [ "$(python3 $BK active $CHIP 2>/dev/null)" != "$OWN" ]; do sleep 60; done
exec 9> "$HOME/spd_qb1_card$CHIP.lock"; flock -w 1800 9 || { log "flock timeout"; exit 3; }
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP TT_BIO_LEASE_HOLDER=$OWN
ARGS="--model opendde --input $FX/cdk2x2_512.yaml --a3m $FX/cdk2x2_512.a3m"
for MODE in normal fast; do
  F=; L=; [ $MODE = fast ] && { F=--fast; L=$FL; }
  P=/dev/shm/odde2-prof-$MODE; rm -rf $P
  log "prof $MODE start"
  ( export TT_METAL_HOME=$M PYTHONPATH=$T:$M/ttnn:$M/tools:$M TT_METAL_PROFILER_DIR=$P ${L:+TT_BIO_LEVERS=$L}; unset LD_LIBRARY_PATH
    timeout 2400 $PY -m tracy -r --no-op-info-cache -o $P --op-support-count 60000 -- \
      $T/perf/mol1/bh/opgap.py prof $ARGS $F --out $OUT/prof-$MODE ) > $OUT/prof-$MODE.log 2>&1 9>&-
  log "prof $MODE rc=$?"
  cp $P/reports/*/ops_perf_results_*.csv $OUT/prof-$MODE/ 2>/dev/null; rm -rf $P
  log "wall $MODE start"
  ( export PYTHONPATH=$T ${L:+TT_BIO_LEVERS=$L}
    timeout 1500 $PY $T/perf/mol1/bh/opgap.py wall $ARGS $F --out $OUT/wall-$MODE ) > $OUT/wall-$MODE.log 2>&1 9>&-
  log "wall $MODE rc=$?"
done
log CENSUS-DONE
