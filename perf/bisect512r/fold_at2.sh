#!/usr/bin/env bash
# fold_at2.sh <label> <commit> -- fold the OpenDDE 512 aa page fixture at <commit>.
# Same fixture/protocol/host as the parent task. Two hard guards the v1 driver lacked:
#   1. checkout is forced and its exit status is checked directly, not through a pipe.
#      v1 piped to `tail` so `||` read tail's status, and four arms silently folded the
#      branch tip after "Please commit your changes... Aborting".
#   2. the JSON's own tt_bio_git is compared with the requested sha after the fold. That is
#      the only guard that cannot be fooled by a checkout that did not happen.
# Lives in an UNTRACKED dir so `git checkout` cannot delete it mid-search.
set -u
WT=/home/moritz/.coworker/wt/opendde-512aa-residual-drift-bisect
PY=/home/moritz/tt-bio/env/bin/python3
OUT=$WT/.bisect-out
LABEL=$1; COMMIT=$2
cd "$WT" || exit 1
WANT=$(git rev-parse "$COMMIT") || exit 1
git checkout -q -f --detach "$WANT"
if [ "$(git rev-parse HEAD)" != "$WANT" ]; then
  echo "CHECKOUT-FAILED $LABEL want=$WANT got=$(git rev-parse HEAD)"; exit 1
fi
rm -f "$OUT/$LABEL.json"
echo "=== $(date -u +%FT%TZ) START $LABEL @ $WANT ==="
echo "FOLD_ENV=[${FOLD_ENV:-}]"
timeout -k 30 900 env ${FOLD_ENV:-} TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
    TT_BIO_LEASE_HOLDER=worker:opendde-512aa-residual-drift-bisect \
    PYTHONPATH=$WT "$PY" "$WT/scripts/gpu_vs_tt/tt_baseline.py" \
    --model opendde --repeat 1 \
    --target perf/size512/fixtures/cdk2x2_512.yaml \
    --msa-a3m perf/size512/fixtures/cdk2x2_512.a3m \
    --label "512 aa" --keep-cif "$OUT/cif_$LABEL" --out "$OUT/$LABEL.json"
rc=$?
[ $rc -eq 124 ] && echo "TIMEOUT $LABEL @ $WANT (900s)"
echo "=== $(date -u +%FT%TZ) END $LABEL rc=$rc ==="
[ -f "$OUT/$LABEL.json" ] || { echo "NOJSON $LABEL @ $WANT"; exit 2; }
"$PY" - "$OUT/$LABEL.json" "$WANT" "$LABEL" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); want=sys.argv[2]; lab=sys.argv[3]
if d.get("tt_bio_git")!=want:
    print(f"SHA-MISMATCH {lab} want={want[:8]} json={d.get('tt_bio_git','?')[:8]}"); sys.exit(3)
p={f.get("plddt") for f in d.get("warm_folds",[])}
s={v for f in d.get("warm_folds",[]) for v in f.get("cif_sha256",{}).values()}
print(f"RESULT {lab} {want[:8]} plddt={sorted(p)} cif={sorted(s)} "
      f"recycles={d.get('recycling_steps')} steps={d.get('sampling_steps')} seed={d.get('seed')}")
PY
