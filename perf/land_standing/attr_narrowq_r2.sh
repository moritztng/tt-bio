#!/bin/bash
# Repeat of the openbind/896 cell (A/A for its digest move) and openfold3/768 with the lever on.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/gate_narrowq/attr
cd "$WT"
export PYTHONPATH="$WT" RELEASE_GATE_CENSUS_PYTHONPATH="$WT" TT_BIO_LEASE_HOLDER=worker:land-standing NQ_TAG=_r2
( TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 "$PY" perf/land_standing/narrowq_cell_ab.py "$OUT/cell_ab_c0_r2.json" \
    openbind:896:on,openbind:896:off,openbind:896:on,openbind:896:off > "$OUT/cell_ab_c0_r2.log" 2>&1
  echo "c0r2 rc=$? $(date -u +%FT%TZ)" >> "$OUT/status3" ) &
( TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=0,2 "$PY" perf/land_standing/narrowq_cell_ab.py "$OUT/cell_ab_c2_r2.json" \
    openfold3:768:on > "$OUT/cell_ab_c2_r2.log" 2>&1
  echo "c2r2 rc=$? $(date -u +%FT%TZ)" >> "$OUT/status3" ) &
wait; echo ALLDONE >> "$OUT/status3"
