#!/usr/bin/env bash
# At and below the 1024-token cap the above-cap fused route is gated out, so flipping the default
# must change neither the bytes nor the time. off/on/off per rung gives both readings in one
# session: the digest answers correctness, and the off-vs-off gap is the floor the off-vs-on gap
# has to be read against. One process per fold -- the flag is never flipped inside a live device
# context (that stack is fetch_queue_reserve_back, and it does not time out).
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-fused-sdpa-default-ship
cd "$WT" || exit 1
OUT=perf/ttx_a3/nochange/quiet
CARD=${CARD:-3}
STEPS=${STEPS:-200}
for size in 298 512 768 1024; do
  for tag_arm in off1:off on:on off2:off; do
    tag="s${size}_${tag_arm%%:*}"; arm="${tag_arm##*:}"
    [ -f "$OUT/$tag.json" ] && { echo "skip $tag"; continue; }
    echo "=== $(date -u +%H:%M:%SZ) $tag arm=$arm load=$(cut -d" " -f1 /proc/loadavg) ==="
    TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
    TT_BIO_LEASE_HOLDER=worker:ttx-a3-fused-sdpa-default-ship \
    timeout 900 /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/ttx_a3/fold_parity_a3.py \
      --dir "$OUT" --arm "$arm" --tag "$tag" --fixture "cdk2x2_${size}" --steps "$STEPS" \
      >> "$OUT/run.log" 2>&1
    rc=$?
    echo "  rc=$rc $( [ -f "$OUT/$tag.json" ] && python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(r[\"fold_s\"],r[\"cif_sha256\"][:16],r[\"triatt_served\"],r[\"triatt_declined\"],r[\"sdpa_picks\"])" "$OUT/$tag.json" )"
  done
done
echo "=== QUIET DONE $(date -u +%H:%M:%SZ) ==="
