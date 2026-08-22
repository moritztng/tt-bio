#!/usr/bin/env bash
# Interleaved fold-level A/B on one card, one process per arm-rep so an env-gated lever is
# resolved at import in both arms. Usage:
#   ob_ab.sh <card> <input-stem> <tag> <reps> <ENV_A=..>,<..> <ENV_B=..>,<..> [extra args]
# Arms alternate A,B,A,B so host drift cancels instead of loading one arm.
set -u
CARD="$1"; STEM="$2"; TAG="$3"; REPS="$4"; ENVA="$5"; ENVB="$6"; shift 6
WT=/home/ttuser/.coworker/wt/openbind-perf-p3
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/openbind/tt_results/ab/$TAG
mkdir -p "$OUT"
cd "$WT" || exit 1
for r in $(seq 1 "$REPS"); do
  for arm in A B; do
    [ "$arm" = A ] && EV="$ENVA" || EV="$ENVB"
    f=$OUT/${STEM}_${arm}${r}.json
    if grep -q device_s_median "$f" 2>/dev/null; then echo "SKIP $TAG $arm$r"; continue; fi
    echo "=== $TAG $STEM arm$arm rep$r card$CARD env[$EV] @ $(date -u +%H:%M:%SZ) ==="
    env PYTHONPATH=$WT TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
        TT_BIO_LEASE_HOLDER=worker:openbind-perf-p3 \
        $(echo "$EV" | tr ',' ' ') \
        "$PY" perf/openbind/tt_ob_run.py --model openbind \
        --input perf/openbind/inputs/$STEM.tt.yaml --repeat 2 \
        --label "${TAG}_${arm}${r}" --out "$f" "$@" 2>&1 | tail -25
    echo "=== rc=$? @ $(date -u +%H:%M:%SZ) ==="
  done
done
echo "AB $TAG COMPLETE @ $(date -u +%H:%M:%SZ)"
