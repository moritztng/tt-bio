#!/bin/bash
# FKBP12 + SB3 on RF3, three atomworks arms x 8 seeds, through `tt-bio predict --model rf3`
# (50 steps, 5 samples, one fresh process per fold), interleaved by seed so drift hits every arm alike.
#   aw221    tree at 0ebfece1d (main's vendored atomworks 2.2.1)
#   aw304    tree at wk/rfd4-atomworks 22f4f52cd (vendored 3.0.4)
#   aw304rot aw304 with 2.2.1's get_random_rots torch branch put back
# Usage: ab8.sh <scratch_root> <card>. Trees live under <scratch_root>/t221, t304, t304rot.
# Accuracy only: no timing is claimed, so AICLK is not sampled.
set -u
S=$1; CARD=$2; OUT=$S/runs
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
declare -A TREE=([aw221]=$S/t221 [aw304]=$S/t304 [aw304rot]=$S/t304rot)
mkdir -p "$OUT"
for seed in 0 1 2 3 4 5 6 7; do
  for arm in aw221 aw304 aw304rot; do
    run=$OUT/${arm}_s${seed}
    [ -e "$run/rc" ] && continue
    for try in 1 2 3 4 5 6; do
      [ -e "$run" ] && mv "$run" "$run.try$(date +%s)"
      mkdir -p "$run"
      echo "$(date -u +%FT%TZ) START $arm s$seed try $try" >> "$OUT/ab.log"
      t0=$(date +%s)
      ( cd "${TREE[$arm]}" && TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
        TT_BIO_LEASE_HOLDER=worker:rfd4-aw-go PYTHONPATH="${TREE[$arm]}" \
        timeout 1800 "$PY" -m tt_bio.main predict "$HERE/fkg_ligand.yaml" --model rf3 \
          --diffusion_samples 5 --seed $seed --msa_dir "$HOME/scratch/rfd4-atomworks/msa" --msa_cache_only \
          --out_dir "$run" ) > "$run/log" 2>&1
      rc=$?
      [ $rc -eq 75 ] || break
      echo "$(date -u +%FT%TZ) CONTENDED $arm s$seed" >> "$OUT/ab.log"; sleep 120
    done
    echo $rc > "$run/rc"
    echo "$(date -u +%FT%TZ) END $arm s$seed rc=$rc $(( $(date +%s)-t0 ))s" >> "$OUT/ab.log"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE" >> "$OUT/ab.log"
