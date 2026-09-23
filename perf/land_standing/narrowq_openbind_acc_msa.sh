#!/bin/bash
# openbind at 896 aa, CLI-default (production) config, lever off vs on at two seeds.
# Same four legs with the fixture's own MSA, because single-sequence openbind sits ~19 A off 1HCL and scores nothing.
# The 6-step census showed the lever moves openbind here; this scores how far at the config users run.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/narrowq_openbind_acc_msa
mkdir -p "$OUT"; cd "$WT"
export PYTHONPATH="$WT" TT_BIO_LEASE_HOLDER=worker:land-standing
leg() { # card seed arm
  local c=$1 s=$2 a=$3 v; [ "$a" = on ] && v=1 || v=0
  TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=0,$c TT_BIO_TRIATT_NARROW_Q_FALLBACK=$v \
    "$PY" -m tt_bio.main predict $OUT/cdk2x2_896_msa.yaml --model openbind \
    --seed "$s" --out_dir "$OUT/s${s}_$a" > "$OUT/s${s}_$a.log" 2>&1
  echo "s$s $a rc=$? $(date -u +%FT%TZ)" >> "$OUT/status"
}
{ leg 0 0 off; leg 0 0 on; } &
{ leg 2 1 off; leg 2 1 on; } &
wait; echo ALLDONE >> "$OUT/status"
