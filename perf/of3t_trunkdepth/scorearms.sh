#!/usr/bin/env bash
# Score every depth against the 0.4.3 float64 reference OF THAT DEPTH, with that depth's own
# pure-bf16 floor and upstream's own autocast recipe scored on the same set.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkdepth
O=/home/ttuser/of3t_trunkdepth
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W" || exit 1
S=perf/of3t_trunkgrad/score_grad.py

for k in 0 8 16 23 32 40 47; do
  echo "=== block $k ALONE ==="
  ARMS=(--arm "DEV_b${k}=$O/DEV_b${k}.pt"
        --arm "FLOOR_PUREBF16=$O/REF_b${k}_bf16pure.pt"
        --arm "UPSTREAM_BF16AUTO=$O/REF_b${k}_bf16auto.pt")
  [ -f "$O/DEV_BREAK_b${k}.pt" ] && ARMS+=(--arm "DEV_BREAK_b${k}=$O/DEV_BREAK_b${k}.pt")
  "$PY" "$S" --reference "$O/REF_b${k}_f64.pt" --intersect \
    --reference-label "upstream 0.4.3, PairFormerBlock $k alone, every parameter and activation float64, its own captured boundary, crop 64" \
    "${ARMS[@]}" --zero-arm A16_ZERO --self-arm INSTRUMENT_FLOOR \
    --note "one block, captured input in, that same block's captured output cotangent back: no chaining" \
    --out "perf/of3t_trunkdepth/SCORE_b${k}.json"
done

echo "=== the cotangent this block hands back, ours against float64 ==="
P=()
for k in 0 8 16 23 32 40 47; do
  P+=(--pair "b${k}=$O/DEV_b${k}_ig.pt:$O/REF_b${k}_f64_ig.pt")
  P+=(--pair "b${k}_FLOOR=$O/REF_b${k}_bf16pure_ig.pt:$O/REF_b${k}_f64_ig.pt")
  [ -f "$O/DEV_BREAK_b${k}_ig.pt" ] && P+=(--pair "b${k}_BREAK=$O/DEV_BREAK_b${k}_ig.pt:$O/REF_b${k}_f64_ig.pt")
done
"$PY" perf/of3t_trunkback/ig_compare.py "${P[@]}" --out perf/of3t_trunkdepth/IG_LADDER.json
echo "SCOREARMS_ALLDONE $(date -u +%FT%TZ)"
