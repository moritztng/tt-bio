#!/bin/bash
# RF3 serving-path gate for the atomworks 3.0.4 bump: `tt-bio predict --model rf3` (50 steps,
# 5 samples, one fresh process per fold), aw-go's protocol and seeds 0-7, three inputs, three arms
# interleaved by seed so drift hits every arm alike:
#   main310  main before the bump (atomworks 2.2.1), py3.10 / biotite 1.2.0, what the engines run today
#   land310  this branch (3.0.4 + biotite 1.2 backport), py3.10 / biotite 1.2.0
#   land312  this branch, py3.12 / biotite 1.6.0 (no backport)
# AICLK is read from sysfs every 2 s while each fold runs (aiclk.txt in the run dir).
# Usage: gate.sh <scratch_root> <card> <main_tree>
set -u
S=$1; CARD=$2; MAIN=$3; OUT=$S/runs
HERE=$(cd "$(dirname "$0")" && pwd)
LAND=$(cd "$HERE/../.." && pwd)
V=$HOME/scratch/shipflow
declare -A TREE=([main310]=$MAIN [land310]=$LAND [land312]=$LAND)
declare -A PY=([main310]=$V/venv-1384f4c1b7bf/bin/python [land310]=$V/venv-1384f4c1b7bf/bin/python
               [land312]=$V/venv312-1384f4c1b7bf/bin/python)
CLK=/sys/class/tenstorrent/tenstorrent!$CARD/tt_aiclk
mkdir -p "$OUT"
for seed in 0 1 2 3 4 5 6 7; do
  for inp in fkg_ligand prot prot_glycan; do
    for arm in main310 land310 land312; do
      run=$OUT/${inp}_${arm}_s${seed}
      [ -e "$run/rc" ] && continue
      for try in 1 2 3 4 5 6; do
        [ -e "$run" ] && mv "$run" "$run.try$(date +%s)"
        mkdir -p "$run"
        echo "$(date -u +%FT%TZ) START $inp $arm s$seed try $try" >> "$OUT/gate.log"
        t0=$(date +%s)
        ( while :; do cat "$CLK"; sleep 2; done ) > "$run/aiclk.txt" 2>/dev/null &
        sampler=$!
        ( cd "${TREE[$arm]}" && TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
          TT_BIO_LEASE_HOLDER=worker:rfd4-land PYTHONPATH="${TREE[$arm]}" \
          timeout 1800 "${PY[$arm]}" -m tt_bio.main predict "$HERE/inputs/$inp.yaml" --model rf3 \
            --diffusion_samples 5 --seed $seed --msa_dir "$S/msa" --msa_cache_only \
            --out_dir "$run" ) > "$run/log" 2>&1
        rc=$?
        kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
        [ $rc -eq 75 ] || break
        echo "$(date -u +%FT%TZ) CONTENDED $inp $arm s$seed" >> "$OUT/gate.log"; sleep 120
      done
      echo $rc > "$run/rc"
      echo "$(date -u +%FT%TZ) END $inp $arm s$seed rc=$rc $(( $(date +%s)-t0 ))s" >> "$OUT/gate.log"
    done
  done
done
echo "$(date -u +%FT%TZ) ALL DONE" >> "$OUT/gate.log"
