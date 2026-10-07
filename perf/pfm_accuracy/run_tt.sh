#!/usr/bin/env bash
# tt-bio Protenix-v2 on the accuracy set, for whichever row holds a leased chip (TT_VISIBLE_DEVICES set by the caller).
# The customer's command line: 5 samples, 200 steps (default), --recycling_steps 10, MSAs from the pc-built cache only.
#   DATA=<data dir> SEEDS="101 102 103" MODES="exact fast" bash run_tt.sh
# Writes $DATA/tt/out/tt_<mode>/<PDB>/seed_<s>/ (tt-bio's own layout) and relays it into
# $DATA/tt/out/tt<mode>_<seeds>/pred/<PDB>/seed_<s>/predictions/<PDB>_sample_<k>.cif, which score.py reads.
set -uo pipefail
DATA=${DATA:?}; HERE=$(cd "$(dirname "$0")" && pwd)
for s in ${SEEDS:-101 102 103}; do
  for mode in ${MODES:-exact fast}; do
    for y in "$DATA"/inputs/*.yaml; do
      p=$(basename "$y" .yaml); o=$DATA/tt/out/tt_$mode/$p/seed_$s
      [ -f "$o/rc" ] && grep -q "rc=0" "$o/rc" && continue
      mkdir -p "$o"
      FAST=(); [ "$mode" = fast ] && FAST=(--fast)
      timeout 3600 tt-bio predict "$y" --model protenix-v2 --recycling_steps 10 --diffusion_samples 5 --seed "$s" \
        --msa_dir "$DATA/msa/cache" --msa_cache_only --out_dir "$o" --override "${FAST[@]}" > "$o/log" 2>&1
      echo "rc=$?" > "$o/rc"; echo "TT-DONE $mode $p $s $(cat "$o/rc")"
    done
  done
done
python3 "$HERE/tt_relayout.py" "$DATA/tt/out" "${SEEDS:-101 102 103}"
