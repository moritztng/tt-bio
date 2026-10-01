#!/bin/bash
# The top of the ladder, after the band walk. The tree serves 896, a bucket above the ceiling
# v0.11.0 documents, so the release needs the tag's answer at the same rung beside it, and the
# first axis that actually refuses on BOTH has still not been found: 928 and 960 go on the tree.
set -u
wt=/home/moritz/.coworker/wt/rel012-verify-bh
tag=$wt/tagtree011
out=$wt/perf/rel012_bh/out/top; mkdir -p "$out"
log=$out/top.log
PY=/home/moritz/bcx_hostcut_venv/bin/python3
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rel012-verify-bh

while ! grep -q "BAND DONE" "$wt/perf/rel012_bh/out/band/band.log" 2>/dev/null; do sleep 20; done

run () {  # run <tag> <tree> <binder>
  local t=$1 tree=$2 b=$3
  cd "$tree"; export PYTHONPATH=$tree
  export JAX_COMPILATION_CACHE_DIR=$out/xlacache_$(basename "$tree")
  echo "=== $(date -u +%FT%TZ) START $t tree=$(git -C "$tree" rev-parse --short HEAD) binder=$b" >> "$log"
  timeout 2400 $PY -u perf/bgx_size/rung.py --params /home/moritz/bcx_shipped/af2_params \
    --out "$out/$t" --target hHSA --binder "$b" --rounds 2 --trajectories 1 \
    > "$out/$t.log" 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) END $t" >> "$log"
}

run g896_tag  "$tag" 270
run t928_auto "$wt"  300
run t960_auto "$wt"  330
echo "=== TOP DONE $(date -u +%FT%TZ)" >> "$log"
