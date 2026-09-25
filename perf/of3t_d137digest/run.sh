#!/usr/bin/env bash
# D137's digest half, re-run on qb1 card 3 (UMD 3 = 0000:c1:00.0 = board ...40aa).
# One model at a time, three arms each, all on the same chip. Sequential: one device context
# per process and one process on the card at a time.
set -u
WT=/home/ttuser/.coworker/wt/of3t-d137digest
cd "$WT" || exit 1
OUT="$WT/perf/of3t_d137digest"
WORK="$WT/perf/of3t_d137digest/work"
mkdir -p "$OUT" "$WORK"

PY=/home/ttuser/tt-bio-dev/env/bin/python3
REPS=${REPS:-3}

export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:of3t-d137digest

for m in openfold3 opendde protenix-v2; do
  echo "=== $m start $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_d137tapegate/inference_ab_with_aa_floor.py \
    --model "$m" \
    --fixture "$WT/perf/size512/fixtures/cdk2x2_128.yaml" \
    --base-tree /tmp/of3t-d137digest-base \
    --tree "$WT" \
    --python "$PY" \
    --workdir "$WORK" \
    --card 3 \
    --reps "$REPS" \
    --out "$OUT/INFERENCE_AB_${m}.json"
  echo "=== $m exit=$? $(date -u +%FT%TZ) ==="
done
echo "=== ALL DONE $(date -u +%FT%TZ) ==="
