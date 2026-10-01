#!/bin/bash
# The paired p150a round: the rel012 integration tree against the v0.11.0 tag, same box, same
# card, same harness interface, sittings ALTERNATING so a host-load drift lands on both arms.
# PD-L1 + a 146 aa binder is 288 tokens, three trajectories: the headline v0.11.0 documents.
# --max-trajectories 6 because the BUDGET caps the interleave: rung.py defaults to 2, so asking
# for three trajectories a card still ran two slots and a first attempt measured a two-way
# interleave labelled as three. v0.11.0's own table used a six-trajectory budget.
# Three sittings an arm. `--rounds` counts gradient rounds across ALL interleaved slots, so 30
# is ten a slot at three trajectories; each slot's first round is dropped as compile-laden,
# leaving at least nine counted rounds a slot an arm. A round is reported PRO RATA: three slots
# interleaved finish three rounds in one slot-to-slot interval, which is how v0.11.0's 6.00 s
# was computed and the only way the two arms compare.
set -u
wt=/home/moritz/.coworker/wt/rel012-verify-bh
tag=$wt/tagtree011
out=$wt/perf/rel012_bh/out/round; mkdir -p "$out"
log=$out/round.log
PY=/home/moritz/bcx_hostcut_venv/bin/python3
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rel012-verify-bh

# One card, one arm at a time: wait out the ceiling walk rather than contend with it.
while ! grep -q "CEILING DONE" "$wt/perf/rel012_bh/out/ceiling/ceiling.log" 2>/dev/null; do
  sleep 30
done

sit () {  # sit <arm> <tree> <n>
  local arm=$1 tree=$2 n=$3
  cd "$tree"
  export PYTHONPATH=$tree
  export JAX_COMPILATION_CACHE_DIR=$out/xlacache_$arm
  echo "=== $(date -u +%FT%TZ) START $arm sitting $n tree=$(git -C "$tree" rev-parse --short HEAD)" >> "$log"
  timeout 3000 $PY -u perf/bgx_size/rung.py --params /home/moritz/bcx_shipped/af2_params \
    --out "$out/${arm}$n" --target hPDL1 --binder 146 --rounds 30 --trajectories 3 --max-trajectories 6 \
    > "$out/${arm}$n.log" 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) END $arm sitting $n" >> "$log"
}

for n in 1 2 3; do
  sit tree "$wt" "$n"
  sit tag011 "$tag" "$n"
done
echo "=== ROUND DONE $(date -u +%FT%TZ)" >> "$log"
