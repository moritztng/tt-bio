#!/usr/bin/env bash
# tt-bio folds for the PVO accuracy set, one model per call, for whichever row holds a leased chip
# (TT_VISIBLE_DEVICES set by the caller). Recycles and steps are the model's own defaults and are not
# passed: opendde runs its 10 recycles, mol1 its 20. Samples are NOT a default we can inherit --
# tt-bio ships 1 for every model but mol1 (main.py:3318), while both upstreams predict 5, so opendde
# is given --diffusion_samples 5 explicitly. A 1-sample TT number against a 5-sample GPU number is a
# 5x cheat in our favour (state/pvo/MODES.md).
# Mol-1 has no graded TT fast mode at these widths, so MODES.md freezes TT fast = TT exact for it:
# run MODES="exact" and report the one mode twice, never a `--fast` run that drops the tuned levers.
#   DATA=~/pvo-accuracy-data MODEL=opendde SEEDS="101 102 103" MODES="exact fast" bash run_tt.sh
#   DATA=~/pvo-accuracy-data MODEL=mol1    SEEDS="101 102 103" MODES="exact"      bash run_tt.sh
# Writes $DATA/<MODEL>/tt/out/tt_<mode>/<PDB>/seed_<s>/ (tt-bio's layout) and relays it into
# $DATA/<MODEL>/tt/out/tt<mode>_<seeds>/pred/<PDB>/seed_<s>/predictions/, which score.py reads.
set -uo pipefail
DATA=${DATA:?}; MODEL=${MODEL:?}; HERE=$(cd "$(dirname "$0")" && pwd)
OUTROOT=$DATA/$MODEL/tt/out
mkdir -p "$OUTROOT"
# AICLK of every node at 1 Hz for the whole run. Accuracy does not depend on it; the log carries it
# anyway, so a fold that ran on a clamped chip can be told apart from one that did not.
( while :; do echo "$(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!*/tt_aiclk 2>/dev/null | tr '\n' ' ')"; sleep 1; done ) \
  >> "$OUTROOT/aiclk.log" & CLK=$!; trap 'kill $CLK 2>/dev/null' EXIT
for s in ${SEEDS:-101 102 103}; do
  for mode in ${MODES:-exact fast}; do
    for y in "$DATA"/inputs/*.yaml; do
      p=$(basename "$y" .yaml); o=$OUTROOT/tt_$mode/$p/seed_$s
      [ -f "$o/rc" ] && grep -q "rc=0" "$o/rc" && continue
      mkdir -p "$o"
      FAST=(); [ "$mode" = fast ] && FAST=(--fast)
      SAMP=(); [ "$MODEL" = opendde ] && SAMP=(--diffusion_samples 5)
      timeout "${FOLD_TIMEOUT:-5400}" tt-bio predict "$y" --model "$MODEL" --seed "$s" \
        --msa_dir "$DATA/msa/cache" --msa_cache_only --out_dir "$o" --override \
        "${SAMP[@]}" "${FAST[@]}" > "$o/log" 2>&1
      echo "rc=$?" > "$o/rc"; echo "TT-DONE $MODEL $mode $p $s $(cat "$o/rc")"
    done
  done
done
python3 "$HERE/tt_relayout.py" "$OUTROOT" "${SEEDS:-101 102 103}"
echo "TT-ALL-DONE $MODEL"
