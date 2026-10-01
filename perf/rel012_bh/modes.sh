#!/bin/bash
# 768 and 832 refused on pc card 0 at the integration tree with memory='auto', which resolves to
# 'fast' at both: 768 fragmented in the forward (151.0 MB asked, 14.2 MB largest block, 2.199 GB
# free) and 832 filled the card in the backward (34.100 of 34.226 GB held by the fold). Both ran
# one trajectory already, so the refusal's own advice was in force.
#
# This matrix decides what v0.12.0 may claim for a p150a. Per axis: does a supported memory mode
# serve it, and what does the v0.11.0 tag do at the same rung on the same card? The tag arm is
# the baseline for "768 used to crash" and runs its own tree's rung.py at its default.
set -u
wt=/home/moritz/.coworker/wt/rel012-verify-bh
tag=$wt/tagtree011
out=$wt/perf/rel012_bh/out/modes; mkdir -p "$out"
log=$out/modes.log
PY=/home/moritz/bcx_hostcut_venv/bin/python3
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rel012-verify-bh

while ! grep -q "CEILING DONE" "$wt/perf/rel012_bh/out/ceiling/ceiling.log" 2>/dev/null; do sleep 20; done

run () {  # run <tag> <tree> <binder> [extra rung.py args...]
  local t=$1 tree=$2 b=$3; shift 3
  cd "$tree"; export PYTHONPATH=$tree
  export JAX_COMPILATION_CACHE_DIR=$out/xlacache_$(basename "$tree")
  echo "=== $(date -u +%FT%TZ) START $t tree=$(git -C "$tree" rev-parse --short HEAD) binder=$b $*" >> "$log"
  timeout 2400 $PY -u perf/bgx_size/rung.py --params /home/moritz/bcx_shipped/af2_params \
    --out "$out/$t" --target hHSA --binder "$b" --rounds 3 --trajectories 1 "$@" \
    > "$out/$t.log" 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) END $t" >> "$log"
}

run t768_lean    "$wt"  150 --memory lean
run t768_offload "$wt"  150 --memory offload
run g768_tag     "$tag" 150
run t832_lean    "$wt"  210 --memory lean
run t832_offload "$wt"  210 --memory offload
run g832_tag     "$tag" 210
run t864_offload "$wt"  240 --memory offload
run g864_tag     "$tag" 240
echo "=== MODES DONE $(date -u +%FT%TZ)" >> "$log"
