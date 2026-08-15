#!/bin/bash
# The §6.1 gate on the two W2 arms.
#
# The blocking fix is claimed BIT-EXACT (perf/relion-full-device-deep-perf/w2_results.md), so the
# gate here is stronger than the standing one: two runs of the same binary differing only in
# TT_COARSE_E should produce byte-identical output star files, not merely a resolution digit that
# rounds the same. A single differing assignment means the claim is wrong.
set -u
S=/home/ttuser/relion-scratch
O=$S/w2/arms
BIN=$S/relion/build-e2e/bin/relion_postprocess
mkdir -p "$S/w2/pp"

echo "=== walls ==="
for a in e1 e16; do
  printf "%-4s " "$a"; cat "$O/$a.time" 2>/dev/null || echo "MISSING"
done

echo
echo "=== [tt_coarse_tune] kernel accounting, per rank ==="
for a in e1 e16; do
  echo "-- $a"; grep -h "tt_coarse_tune] E=" "$O/$a.log" 2>/dev/null
done

echo
echo "=== bit-exactness of the answer: sha256 of every iteration's data.star ==="
for it in 013 014 015 016 017; do
  f1=$O/e1_run_it${it}_data.star; f2=$O/e16_run_it${it}_data.star
  [ -f "$f1" ] && [ -f "$f2" ] || { echo "it$it  MISSING"; continue; }
  s1=$(sha256sum < "$f1" | cut -c1-16); s2=$(sha256sum < "$f2" | cut -c1-16)
  [ "$s1" = "$s2" ] && v=IDENTICAL || v="DIFFERS"
  echo "it$it  $s1  $s2  $v"
done

echo
echo "=== half maps ==="
for a in e1 e16; do
  for h in 1 2; do
    f=$O/${a}_run_half${h}_class001_unfil.mrc
    [ -f "$f" ] && echo "$a half$h $(sha256sum < "$f" | cut -c1-16)" || echo "$a half$h MISSING"
  done
done

echo
echo "=== relion_postprocess, unmasked, --angpix 1.244835 (the standing recipe) ==="
for a in e1 e16; do
  h1=$O/${a}_run_half1_class001_unfil.mrc
  h2=$O/${a}_run_half2_class001_unfil.mrc
  [ -f "$h1" ] || { echo "$a: MISSING $h1"; continue; }
  cp -f "$h1" "$S/w2/pp/w2_${a}_half1_class001_unfil.mrc"
  cp -f "$h2" "$S/w2/pp/w2_${a}_half2_class001_unfil.mrc"
  "$BIN" --i "$S/w2/pp/w2_${a}_half1_class001_unfil.mrc" --angpix 1.244835 \
         --o "$S/w2/pp/pp_w2_${a}" > "$S/w2/pp/w2_${a}.log" 2>&1
  printf "%-4s rc=%s  " "$a" "$?"
  grep -E "_rlnFinalResolution" "$S/w2/pp/pp_w2_${a}.star" 2>/dev/null | tail -1
done

echo
echo "=== the refinement's own reported resolution (last iteration) ==="
for a in e1 e16; do
  printf "%-4s " "$a"
  grep -hoE "Estimated resolution = *[0-9.]+" "$O/$a.log" 2>/dev/null | tail -1
done
