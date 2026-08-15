#!/bin/bash
# relion_postprocess on this leg's arms, in the e2e leg's recipe byte for byte, so the numbers sit
# beside its 4.033896 A. Unmasked, --angpix 1.244835, no B-factor, and the e2e build's binary: the
# bridge does not touch reconstruction, so using build-e2e keeps the comparison to one variable.
#
# relion_postprocess finds half2 from the half1 filename, so the maps are copied to pp/ first.
#
#   bash fine_postprocess.sh check tt
set -u
S=/home/ttuser/relion-scratch
BIN=$S/relion/build-e2e/bin/relion_postprocess
mkdir -p "$S/pp"
for arm in "$@"; do
  h1=$S/fine/${arm}_run_half1_class001_unfil.mrc
  h2=$S/fine/${arm}_run_half2_class001_unfil.mrc
  if [ ! -f "$h1" ]; then echo "$arm: MISSING $h1 (did the arm converge?)"; continue; fi
  cp -f "$h1" "$S/pp/fine_${arm}_half1_class001_unfil.mrc"
  cp -f "$h2" "$S/pp/fine_${arm}_half2_class001_unfil.mrc"
  "$BIN" --i "$S/pp/fine_${arm}_half1_class001_unfil.mrc" --angpix 1.244835 \
         --o "$S/pp/pp_fine_${arm}" > "$S/pp/fine_${arm}.log" 2>&1
  echo "== $arm rc=$?"
  grep -E 'FINAL RESOLUTION|masking' "$S/pp/fine_${arm}.log"
  grep -E '_rlnFinalResolution' "$S/pp/pp_fine_${arm}.star" 2>/dev/null
done
