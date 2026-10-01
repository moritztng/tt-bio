#!/bin/bash
# The 768 design axis refused on pc card 0 at the integration tree with memory='auto', which
# resolved to 'fast': round 1 completed in 106.1 s and round 2's forward asked for 151.0 MB in
# one DRAM buffer against a largest contiguous block of 14.2 MB. 2.199 GB was free, so this is
# fragmentation and not a full card.
#
# Three rungs decide what the release may say about 768 on a p150a:
#   lean / offload  -- does a supported memory mode serve the axis the auto mode refused?
#   tag             -- what the v0.11.0 tag does at the same rung on the same card, which is the
#                      baseline the claim "768 used to crash" is measured against.
set -u
wt=/home/moritz/.coworker/wt/rel012-verify-bh
tag=$wt/tagtree011
out=$wt/perf/rel012_bh/out/followup768; mkdir -p "$out"
log=$out/followup.log
PY=/home/moritz/bcx_hostcut_venv/bin/python3
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rel012-verify-bh

while ! grep -q "ROUND DONE" "$wt/perf/rel012_bh/out/round/round.log" 2>/dev/null; do sleep 30; done

run () {  # run <tag> <tree> [extra rung.py args...]
  local t=$1 tree=$2; shift 2
  cd "$tree"; export PYTHONPATH=$tree
  export JAX_COMPILATION_CACHE_DIR=$out/xlacache_$(basename "$tree")
  echo "=== $(date -u +%FT%TZ) START $t tree=$(git -C "$tree" rev-parse --short HEAD) $*" >> "$log"
  timeout 3000 $PY -u perf/bgx_size/rung.py --params /home/moritz/bcx_shipped/af2_params \
    --out "$out/$t" --target hHSA --binder 150 --rounds 3 --trajectories 1 "$@" \
    > "$out/$t.log" 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) END $t" >> "$log"
}

run lean   "$wt" --memory lean
run offload "$wt" --memory offload
run tag011 "$tag"
run auto2  "$wt"
echo "=== FOLLOWUP768 DONE $(date -u +%FT%TZ)" >> "$log"
