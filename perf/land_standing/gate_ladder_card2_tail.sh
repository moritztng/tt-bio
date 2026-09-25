#!/bin/bash
# Third size-ladder slice for the narrow-q tip: the five models the wedged 01:50Z run folded but
# never scored. Waits for gate_ladder_card2_slices.sh to release benchlock, then runs on card 2.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/gate_narrowq
cd "$WT" || exit 1
exec 9>/home/ttuser/.coworker/state/benchlock.flock
flock -w 7200 9 || { echo "benchlock not free after 2h; not starting" >> "$OUT/card2_slices.log"; exit 3; }
grep -q "could not be reset" "$OUT/sl_opendde_nesso1_openbind.log" && { echo "card 2 dirty; tail slice not started" >> "$OUT/card2_slices.log"; exit 4; }
export PYTHONPATH="$WT" RELEASE_GATE_CENSUS_PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=0,2 TT_BIO_LEASE_HOLDER=worker:land-standing
slice=boltz2,esmfold2,protenix-v1,protenix-v2,openfold3
echo "######## slice $slice tip $(git rev-parse --short HEAD) $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg) ########" >> "$OUT/card2_slices.log"
"$PY" scripts/release_gate.py --model size-ladder --size-ladder-models "$slice" --keep \
    --journal "$OUT/journal_sl_first5.json" > "$OUT/sl_first5.log" 2>&1
echo "slice $slice rc=$? $(date -u +%FT%TZ)" >> "$OUT/card2_slices.log"
