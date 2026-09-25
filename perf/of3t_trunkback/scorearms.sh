#!/usr/bin/env bash
# Score every arm against the 0.4.3 float64 reference of its OWN scope. Two scopes:
# the 48-block stack the predecessor measured, and each captured boundary as one block.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkback
O=/home/ttuser/of3t_trunkback
P=/home/ttuser/of3t_trunkgrad
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
S=perf/of3t_trunkgrad/score_grad.py

echo "=== 48-stack, each arm on its OWN reached set ==="
"$PY" "$S" --reference "$O/REF043_F64_48.pt" \
  --reference-label "upstream 0.4.3, 48 PairFormerBlock, every parameter and activation float64, crop 64" \
  --arm "DEV_SHIPPED_48=$O/DEV_SHIPPED_48.pt" \
  --arm "DEV_BREAK_48=$O/DEV_BREAK_48.pt" \
  --arm "REF043_BF16_48=$O/REF043_BF16_48.pt" \
  --arm "DEV_SHIPPED_PREDECESSOR=$P/DEV_SHIPPED.pt" \
  --arm "DEV_SCALED_PREDECESSOR=$P/DEV_SCALED.pt" \
  --zero-arm A16_ZERO --self-arm INSTRUMENT_FLOOR \
  --note "own-set scoring; DEV_*_PREDECESSOR are of3t-trunkgrad dumps re-scored here" \
  --out perf/of3t_trunkback/SCORE_48_ownset.json

echo "=== 48-stack, shared denominator over the three arms this row ran ==="
"$PY" "$S" --reference "$O/REF043_F64_48.pt" --intersect \
  --reference-label "upstream 0.4.3, 48 PairFormerBlock, float64, crop 64" \
  --arm "DEV_SHIPPED_48=$O/DEV_SHIPPED_48.pt" \
  --arm "DEV_BREAK_48=$O/DEV_BREAK_48.pt" \
  --arm "REF043_BF16_48=$O/REF043_BF16_48.pt" \
  --zero-arm A16_ZERO --self-arm INSTRUMENT_FLOOR \
  --note "shared set over DEV_SHIPPED_48, DEV_BREAK_48, REF043_BF16_48" \
  --out perf/of3t_trunkback/SCORE_48_shared.json

for k in 0 23 47; do
  echo "=== block $k ALONE ==="
  "$PY" "$S" --reference "$O/REF_b${k}_f64.pt" --intersect \
    --reference-label "upstream 0.4.3, PairFormerBlock $k alone, float64, its own captured boundary, crop 64" \
    --arm "DEV_b${k}=$O/DEV_b${k}.pt" \
    --arm "REF_b${k}_BF16=$O/REF_b${k}_bf16auto.pt" \
    --zero-arm A16_ZERO --self-arm INSTRUMENT_FLOOR \
    --note "one block, captured input in, that same block's captured output cotangent back: no chaining" \
    --out "perf/of3t_trunkback/SCORE_b${k}.json"
done
echo "SCOREARMS_ALLDONE $(date -u +%FT%TZ)"
