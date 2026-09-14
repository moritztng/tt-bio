#!/usr/bin/env bash
# The A/A control for run_sizes.sh. The arms there are bit-identical at every rung, so any
# wall-clock gap between them is host noise; this measures that noise in the same session by
# repeating the SAME arm. Without it "no perf delta below the cap" is an assertion.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-fused-sdpa-default-ship
cd "$WT" || exit 1
OUT=perf/ttx_a3/nochange
CARD=${CARD:-3}
STEPS=${STEPS:-200}
for size in 298 512 768 1024; do
  tag="s${size}_off2"
  [ -f "$OUT/$tag.json" ] && { echo "skip $tag (done)"; continue; }
  echo "=== $(date -u +%H:%M:%SZ) $tag ==="
  TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
  TT_BIO_LEASE_HOLDER=worker:ttx-a3-fused-sdpa-default-ship \
  timeout 1800 /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/ttx_a3/fold_parity_a3.py \
    --dir "$OUT" --arm off --tag "$tag" --fixture "cdk2x2_${size}" --steps "$STEPS" \
    >> "$OUT/run.log" 2>&1
  echo "  rc=$? $( [ -f "$OUT/$tag.json" ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(r[\"fold_s\"],r[\"cif_sha256\"][:16])" "$OUT/$tag.json" )"
done
echo "=== AA DONE $(date -u +%H:%M:%SZ) ==="
