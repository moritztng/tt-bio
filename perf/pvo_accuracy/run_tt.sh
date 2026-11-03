#!/usr/bin/env bash
# tt-bio folds for the PVO accuracy set, one model per call, for whichever row holds a leased chip
# (TT_VISIBLE_DEVICES set by the caller). Every sampling setting is the model's own default --
# samples, steps and recycles are NOT passed, so opendde runs its 10 recycles and mol1 its 20/5,
# which is what the charter asks to compare ("the model's default samples/steps/recycles").
#   DATA=~/pvo-accuracy-data MODEL=opendde SEEDS="101 102 103" MODES="exact fast" bash run_tt.sh
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
      timeout "${FOLD_TIMEOUT:-5400}" tt-bio predict "$y" --model "$MODEL" --seed "$s" \
        --msa_dir "$DATA/msa/cache" --msa_cache_only --out_dir "$o" --override "${FAST[@]}" > "$o/log" 2>&1
      echo "rc=$?" > "$o/rc"; echo "TT-DONE $MODEL $mode $p $s $(cat "$o/rc")"
    done
  done
done
python3 "$HERE/tt_relayout.py" "$OUTROOT" "${SEEDS:-101 102 103}"
echo "TT-ALL-DONE $MODEL"
