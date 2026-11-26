#!/bin/bash
# INFERENCE-AB for r4t-land: RFD4 design, grad off, on one qb2 chip.
#   A/A floor  = the SAME sha (origin/main) run twice on the same chip.
#   A/B        = wk/r4t-step and the merged landing tree wk/r4t-land against that floor.
# Every leg writes design.py's own record (aiclk sampled during the fold) plus a sha256 of the CIF.
#   bash r4t_land_ab.sh <chip> <outdir>
set -u
N=${1:?chip}; R=${2:?outdir}
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
T=$HOME/r4t/land/trees; IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle
BOOK="python3 $HOME/spd/feed/book.py"; OWN=fleet-r4t-land
mkdir -p "$R"; R=$(cd "$R" && pwd)
stamp(){ echo "$1 $(date -u +%FT%TZ)" >> "$R/done"; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:r4t-land
stamp "booked $($BOOK add $OWN $N 21600 --now --note 'r4t-land INFERENCE-AB')"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N 21600 --now >/dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
clear=0
for _ in $(seq 360); do
  if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q "/dev/tenstorrent/$N\$"; then clear=0; else clear=$((clear+1)); fi
  [ $clear -ge 3 ] && break; sleep 10
done
[ $clear -ge 3 ] || { stamp "chip $N never free"; exit 1; }

run_leg(){ # leg sha
  local leg=$1 sha=$2
  git -C "$T" checkout -q --detach "$sha" || { stamp "$leg checkout FAIL"; return 1; }
  cd "$T" || return 1
  PYTHONPATH=$T:$T/perf timeout 3600 $PY perf/rfd4/e2e/design.py --spec $IN/pdl1-128.yaml \
     --weights $W --out "$R/$leg" --seed 0 --designs 2 --rng upstream > "$R/$leg.log" 2>&1
  stamp "$leg sha=$(git -C "$T" rev-parse --short HEAD) rc=$?"
  sha256sum "$R/$leg"/*_generated.cif >> "$R/digests.txt" 2>&1
  grep -o '"aiclk[^,]*' "$R/$leg.log" | tail -4 >> "$R/aiclk.txt"
}
MAIN=$(git -C "$T" rev-parse origin/main)
run_leg aa1 "$MAIN"
run_leg aa2 "$MAIN"
run_leg ab_step "$(git -C "$T" rev-parse origin/wk/r4t-step)"
run_leg ab_land "$(git -C "$T" rev-parse origin/wk/r4t-land)"
for leg in aa2 ab_step ab_land; do
  a=$(ls "$R/aa1"/*_generated.cif 2>/dev/null | head -1); b=$(ls "$R/$leg"/*_generated.cif 2>/dev/null | head -1)
  [ -n "$a" ] && [ -n "$b" ] && PYTHONPATH=$T $PY perf/rfd4_land/cif_compare.py "$b" "$a" --json "$R/cmp-$leg.json" >> "$R/cmp.txt" 2>&1
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
