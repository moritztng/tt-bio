#!/bin/bash
# RF3 serving-path A/B: atomworks 2.2.1 (merge-base tree) vs 3.0.4 (this branch), same inputs, same
# seeds, `tt-bio predict --model rf3` (RF3 defaults: 50 steps), 5 samples, one fresh process per fold.
# Both arms run in the same py3.12 venv (biotite 1.6.0): 3.0.4 cannot import on py3.10 (see ATOMWORKS.md).
# Usage: START=<epoch> ab.sh <out_root> <card>
# Waits for START (the book.py window), then for the card to have no open handle for 60 s before
# every fold, and retries a fold the device lease refuses (rc 75) instead of fighting for the card.
set -u
OUT=$1; CARD=$2; START=${START:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
S=$HOME/scratch/rfd4-atomworks
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
declare -A TREE=([aw221]=$S/main-tree [aw304]=$(cd "$HERE/../.." && pwd))
mkdir -p "$OUT"
wait_idle() {
  local q=0
  while [ $q -lt 4 ]; do
    if fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; then q=0; else q=$((q+1)); fi
    sleep 15
  done
}
while [ "$(date +%s)" -lt "$START" ]; do sleep 60; done
for inp in prot prot_glycan fkg_ligand; do
  for seed in 0 1; do
    for arm in aw221 aw304; do
      run=$OUT/${inp}_${arm}_s${seed}
      [ -e "$run/rc" ] && continue
      for try in 1 2 3 4 5 6 7 8 9 10; do
        rm -rf "$run"; mkdir -p "$run"
        wait_idle
        echo "$(date -u +%FT%TZ) START $inp $arm s$seed try $try" >> "$OUT/ab.log"
        t0=$(date +%s)
        ( cd "${TREE[$arm]}" && TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=2,$CARD \
          TT_BIO_LEASE_HOLDER=worker:rfd4-atomworks PYTHONPATH="${TREE[$arm]}" \
          timeout 2400 "$PY" -m tt_bio.main predict "$HERE/inputs/$inp.yaml" --model rf3 \
            --diffusion_samples 5 --seed $seed --msa_dir "$S/msa" --msa_cache_only \
            --out_dir "$run" ) > "$run/log" 2>&1
        rc=$?
        [ $rc -eq 75 ] || break
        echo "$(date -u +%FT%TZ) CONTENDED $inp $arm s$seed" >> "$OUT/ab.log"; sleep 120
      done
      echo $rc > "$run/rc"
      echo "$(date -u +%FT%TZ) END $inp $arm s$seed rc=$rc $(( $(date +%s)-t0 ))s" >> "$OUT/ab.log"
    done
  done
done
echo "$(date -u +%FT%TZ) ALL DONE" >> "$OUT/ab.log"
