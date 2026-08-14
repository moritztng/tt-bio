#!/bin/bash
# relion_postprocess on both arms, in the recipe the one-iteration comparison already used, so the
# numbers are comparable to the 3.50195 A it recorded.
#
# That recipe is UNMASKED ("== Not performing any masking ..." in pp/A_ref.log) with --angpix
# 1.244835 and no B-factor. The tutorial's own mask exists at Tutorial5.0/MaskCreate/job020/mask.mrc
# and is deliberately not used: a masked FSC would be a different, higher number and would not sit
# beside anything else in this lineage.
#
# relion_postprocess finds half2 from the half1 filename, so the maps are copied to pp/<stem>_half*
# first, which is also what the earlier run did.
set -u
S=/home/ttuser/relion-scratch
BIN=$S/relion/build-e2e/bin/relion_postprocess
mkdir -p "$S/pp"
for arm in ref tt; do
  h1=$S/e2e/${arm}_run_half1_class001_unfil.mrc
  h2=$S/e2e/${arm}_run_half2_class001_unfil.mrc
  if [ ! -f "$h1" ]; then echo "$arm: MISSING $h1 (did the arm converge?)"; continue; fi
  cp -f "$h1" "$S/pp/e2e_${arm}_half1_class001_unfil.mrc"
  cp -f "$h2" "$S/pp/e2e_${arm}_half2_class001_unfil.mrc"
  "$BIN" --i "$S/pp/e2e_${arm}_half1_class001_unfil.mrc" --angpix 1.244835 \
         --o "$S/pp/pp_e2e_${arm}" > "$S/pp/e2e_${arm}.log" 2>&1
  echo "== $arm rc=$?"
  grep -E "FINAL RESOLUTION|masking" "$S/pp/e2e_${arm}.log"
  grep -E "_rlnFinalResolution" "$S/pp/pp_e2e_${arm}.star" 2>/dev/null
done
