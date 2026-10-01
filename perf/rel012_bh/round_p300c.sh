#!/bin/bash
# The same paired 288-token round on qb2 card 0, a p300c. This is the OPTIONAL leg: bcp-evo and
# bcw-callcut were graded on a p300c and it is worth confirming at the merge tree, but it is a
# DIFFERENT BOARD from the p150a the release quotes and its number never joins that trail.
# Card 0's lease was released by rel012-integrate at 22:21Z and the box rebooted an hour ago;
# this run reacquires it as worker:rel012-verify-bh through tt-bio's own lease write.
set -u
wt=/home/ttuser/.coworker/wt/rel012-verify-bh
tag=$wt/tagtree011
out=$wt/perf/rel012_bh/out/round_p300c; mkdir -p "$out"
log=$out/round.log
PY=/home/ttuser/bcx_e2e_venv/bin/python3
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rel012-verify-bh

setsid nohup $PY $wt/perf/rel012_bh/aiclk_sampler.py "$out/aiclk.tsv" >/dev/null 2>&1 < /dev/null &
clk=$!

sit () {  # sit <arm> <tree> <n>
  local arm=$1 tree=$2 n=$3
  cd "$tree"
  export PYTHONPATH=$tree
  export JAX_COMPILATION_CACHE_DIR=$out/xlacache_$arm
  echo "=== $(date -u +%FT%TZ) START $arm sitting $n tree=$(git -C "$tree" rev-parse --short HEAD)" >> "$log"
  timeout 3000 $PY -u perf/bgx_size/rung.py --params /home/ttuser/bcx_e2e/af2_params \
    --out "$out/${arm}$n" --target hPDL1 --binder 146 --rounds 4 --trajectories 3 \
    > "$out/${arm}$n.log" 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) END $arm sitting $n" >> "$log"
}

for n in 1 2 3; do
  sit tree "$wt" "$n"
  sit tag011 "$tag" "$n"
done
kill $clk 2>/dev/null
echo "=== ROUND_P300C DONE $(date -u +%FT%TZ)" >> "$log"
