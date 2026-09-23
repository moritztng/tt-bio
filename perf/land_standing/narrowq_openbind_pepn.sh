#!/bin/bash
# openbind on PepN (3B34, 891 aa -> 896 padded), CLI defaults with a ColabFold MSA fetched once and
# shared, narrow-q off vs on at seeds 0-2. cdk2x2 could not score the lever (19 A off 1HCL
# single-sequence, chaotic with an MSA); this is a natural monomer openbind should fold confidently.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/narrowq_openbind_pepn
mkdir -p "$OUT"; cd "$WT"
export PYTHONPATH="$WT" TT_BIO_LEASE_HOLDER=worker:land-standing
leg() { # card seed arm
  local c=$1 s=$2 a=$3 v; [ "$a" = on ] && v=1 || v=0
  TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=0,$c TT_BIO_TRIATT_NARROW_Q_FALLBACK=$v \
    "$PY" -m tt_bio.main predict perf/land_standing/fixtures/pepn_3b34.yaml --model openbind \
    --msa_dir "$OUT/msa" --seed "$s" --out_dir "$OUT/s${s}_$a" > "$OUT/s${s}_$a.log" 2>&1
  echo "s$s $a rc=$? $(date -u +%FT%TZ)" >> "$OUT/status"
}
leg 0 0 off
{ leg 0 0 on; leg 0 1 off; leg 0 1 on; } &
{ leg 2 2 off; leg 2 2 on; } &
wait; echo ALLDONE >> "$OUT/status"
