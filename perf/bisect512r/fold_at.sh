#!/usr/bin/env bash
# fold_at.sh <label> [<commit>] — fold the OpenDDE 512 aa page fixture and print plDDT + CIF sha.
# Adapted from perf/bisect512/fold_at.sh (parent task opendde-512aa-numerics-drift-bisect),
# same protocol, same fixture, same host/card. Each arm runs that commit's OWN tt_baseline.py,
# because the harness changed inside this range and a pinned new harness will not import on the
# old tree. No benchlock: this measures OUTPUT, not time.
set -u
WT=/home/moritz/.coworker/wt/opendde-512aa-residual-drift-bisect
PY=/home/moritz/tt-bio/env/bin/python3
OUT=$WT/.bisect-out
LABEL=$1
COMMIT=${2:-}
mkdir -p "$OUT"
cd "$WT" || exit 1
if [ -n "$COMMIT" ]; then
  git checkout -q --detach "$COMMIT" 2>&1 | tail -2 || exit 1
fi
SHA=$(git rev-parse HEAD)
echo "=== $(date -u +%FT%TZ) START $LABEL @ $SHA ==="
timeout -k 30 900 env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
    TT_BIO_LEASE_HOLDER=worker:opendde-512aa-residual-drift-bisect \
    PYTHONPATH=$WT "$PY" "$WT/scripts/gpu_vs_tt/tt_baseline.py" \
    --model opendde --repeat "${REPEAT:-1}" \
    --target perf/size512/fixtures/cdk2x2_512.yaml \
    --msa-a3m perf/size512/fixtures/cdk2x2_512.a3m \
    --label "512 aa" --keep-cif "$OUT/cif_$LABEL" --out "$OUT/$LABEL.json"
rc=$?
[ $rc -eq 124 ] && echo "TIMEOUT $LABEL @ $SHA (900s) -- pre-08-27 pair-cond 512aa deadlock territory"
echo "=== $(date -u +%FT%TZ) END $LABEL rc=$rc ==="
"$PY" - "$OUT/$LABEL.json" "$SHA" "$LABEL" <<'PY' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]))
for f in d.get("warm_folds",[]):
    print(f"RESULT {sys.argv[3]} {sys.argv[2][:8]} plddt={f.get('plddt')} sha={list(f.get('cif_sha256',{}).values())} recycles={d.get('recycling_steps')}")
PY
