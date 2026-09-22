#!/usr/bin/env bash
# The gate commit ALONE. 6d7f32dc0 is exactly 7f386f768^, so base-vs-this-tree is the gate
# commit and nothing else -- where the main run's `off` arm is the whole 130-commit
# composition, which also carries 701ddcf63 (the openfold3 trunk sqrt(24) pair-bias fix) and
# therefore moves the openfold3 digest for a reason that has nothing to do with the gate.
set -u
WT=/home/ttuser/.coworker/wt/of3t-d137digest
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3

while kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "=== main run pid $1 gone, gate isolation start $(date -u +%FT%TZ) ==="

export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:of3t-d137digest

for m in openfold3 opendde; do
  "$PY" perf/of3t_d137tapegate/inference_ab_with_aa_floor.py \
    --model "$m" \
    --fixture "$WT/perf/size512/fixtures/cdk2x2_128.yaml" \
    --base-tree /tmp/of3t-d137digest-base \
    --tree /tmp/of3t-d137digest-gate \
    --python "$PY" \
    --workdir "$WT/perf/of3t_d137digest/work_gate" \
    --card 3 \
    --reps 3 \
    --out "$WT/perf/of3t_d137digest/GATE_COMMIT_AB_${m}.json"
  echo "=== gate-isolation $m exit=$? $(date -u +%FT%TZ) ==="
done
echo "=== GATE ISOLATION DONE $(date -u +%FT%TZ) ==="
