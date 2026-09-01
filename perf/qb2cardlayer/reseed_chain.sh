#!/usr/bin/env bash
# reseed_chain.sh — the seven card-layer cells whose draws are complete.
# One benchlock acquisition for the whole chain, not one per model: benchlock is host-scoped with
# no fairness (benchlock-host-scoped-no-fairness-starves-waiters), so seven separate waits behind
# a 9.5-minute boltzgen draw would starve. Exclusivity is what matters for the measurement and one
# lock gives it. Still one perf_regression.py --update-baseline call per model, so each cell keeps
# its own note; a multi-model update stamps one note over the whole block.
set -u
WT=/home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed
O=$WT/perf/qb2cardlayer
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH="$WT" TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:qb2-card-layer-baseline-reseed
PY=/home/ttuser/tt-bio-dev/env/bin/python

WHY="Cell was stale LOW, the direction that hides a regression, and the gap predates the 2026-09-01 qb2 hardware swap: the retired box read %s at v0.7.2 on 2026-08-28, %s from this reseed. Draws perf/qb2cardlayer/draws.tsv, see state/qb2-card-layer-baseline-reseed.md and state/qb2-new-hardware-baseline-crosscheck.md."
HOW="Reseeded 2026-09-01 on the replacement qb2, p300c card 0, benchlocked, one fresh process per draw: %d draws, median %s %s, band %s. This cell is the reseed run's own draw, not the median."

note() { printf "$HOW $WHY" "$2" "$3" "$4" "$5" "$6" "$7"; }

run() { # run <model> <n> <median> <unit> <band> <oldbox> <delta>
  local m=$1
  local n; n=$(note "$@")
  echo "=== $m ==="
  $PY "$WT/scripts/perf_regression.py" --model "$m" --update-baseline --note "$n" 2>&1 | grep -E "^\[|Wrote"
  python3 "$O/promote.py" "$m"
}

cd "$WT"
{
run boltz2          5 1.772869   structures/s 1.5%  "1.7632 structures/s"   "+0.5%"
run esmfold2        5 3.214906   structures/s 2.6%  "3.2347 structures/s"   "-0.6%"
run esmfold2-fast   5 3.890141   structures/s 2.5%  "3.8323 structures/s"   "+1.5%"
run esmc-300m       5 147.199459 seq/s        2.5%  "150.4193 seq/s"        "-2.1%"
run esmc-600m       5 109.088683 seq/s        2.4%  "107.3617 seq/s"        "+1.6%"
run boltz2-affinity 5 0.023954   affinities/s 0.8%  "0.02294 affinities/s"  "+4.4%"
echo "=== esmc-6b ==="
$PY "$WT/scripts/perf_regression.py" --model esmc-6b --update-baseline --note "Reseeded 2026-09-01 on the replacement qb2, p300c card 0, benchlocked, one fresh process per draw: 7 draws, median 10.492650 seq/s, band 1.1%. This cell is the reseed run own draw, not the median. The old cell 4.749028 dated 2026-07-19 was 121% low, the direction that hides a regression. FRESH BASELINE WITH NO HARDWARE-SWAP CONTROL, and there never can be one: esmc-6b was excluded from the v0.7.2 gate on cost, so the retired box left no same-tag reading, and it has been wiped and shipped back. The indirect evidence is its two siblings esmc-300m and esmc-600m, same family and same protocol, which reproduce the retired box to 2.1%. Draws perf/qb2cardlayer/draws.tsv, see state/qb2-card-layer-baseline-reseed.md." 2>&1 | grep -E "^\[|Wrote"
python3 "$O/promote.py" esmc-6b
echo "=== chain done $(date -u +%FT%TZ) ==="
} 
