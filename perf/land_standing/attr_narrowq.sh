#!/bin/bash
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/gate_narrowq/attr
cd "$WT"
export PYTHONPATH="$WT" RELEASE_GATE_CENSUS_PYTHONPATH="$WT" TT_BIO_LEASE_HOLDER=worker:land-standing
( TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 "$PY" perf/land_standing/narrowq_cell_ab.py "$OUT/cell_ab_c0.json" \
    protenix-v2:896:off,protenix-v2:896:on,openbind:896:off,openbind:896:on,openbind:640:off,openbind:640:on > "$OUT/cell_ab_c0.log" 2>&1
  echo "c0 rc=$? $(date -u +%FT%TZ)" >> "$OUT/status2" ) &
( TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=0,2 "$PY" perf/land_standing/narrowq_cell_ab.py "$OUT/cell_ab_c2.json" \
    openfold3:768:off > "$OUT/cell_ab_c2.log" 2>&1
  echo "c2 rc=$? $(date -u +%FT%TZ)" >> "$OUT/status2" ) &
wait; echo ALLDONE >> "$OUT/status2"
