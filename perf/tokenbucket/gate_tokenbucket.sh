#!/usr/bin/env bash
# Step G of the token-bucket flip: the release gate on every leg the flip touches, plus the
# off-64 diagnostic rung the standard size ladder cannot see (256/512/640/768 are all multiples
# of 64, so the ladder is structurally blind to this lever).
#
# Every leg below runs at an unaligned token count, so every leg WILL move. That is the fix
# working. The external reference fixtures are an upstream run and are never regenerated; only
# device-side drift JSONs get re-baselined, in this branch, with the reason in the commit.
#
#     CARD=<your launch grant> bash perf/tokenbucket/gate_tokenbucket.sh
set -u
: "${CARD:?set CARD to this launch grant}"
WT=/home/ttuser/.coworker/wt/protenix-opendde-token-bucket-flip-measure
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:protenix-opendde-token-bucket-flip-measure"
OUT=perf/tokenbucket/gate
mkdir -p $OUT

if ! timeout 300 env $LEASE PYTHONPATH=$WT "$PY" -u perf/tokenbucket/preflight_card.py \
     >/dev/null 2>&1; then
  echo "PREFLIGHT FAILED on card $CARD. Not gating."; exit 1
fi
echo "$(date -Is) preflight OK on card $CARD"

run() {  # run <tag> <cmd...>; rc is the command's, not an echo's (pass 2 lost four runs to that)
  tag=$1; shift
  if [ -f "$OUT/$tag.done" ]; then echo "SKIP $tag (already done)"; return 0; fi
  echo "=== $(date -Is) BEGIN $tag"
  env $LEASE PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm "$@" > "$OUT/$tag.log" 2>&1
  rc=$?
  echo "=== $(date -Is) END $tag rc=$rc"
  [ $rc -eq 0 ] && touch "$OUT/$tag.done"
  return $rc
}

run parity "$PY" -u scripts/full_parity_gate.py \
  --leg protenix-prot-msa --leg protenix-ubq-msa --leg protenix-hsa-msa --leg protenix-9ncy-msa \
  --leg opendde-trpcage-nomsa --leg opendde-prot-prod --leg opendde-abag --leg capacity
run perf "$PY" -u scripts/perf_regression.py
run sizeladder "$PY" -u scripts/release_gate.py --model size-ladder

# The diagnostic rung. 298 is not a multiple of 64, so this is the only ladder run that can see
# the bucket at all. Census only: report it, never re-record off it.
export RELEASE_GATE_SIZE_RUNGS=298
run sizeladder298 "$PY" -u scripts/release_gate.py --model size-ladder
unset RELEASE_GATE_SIZE_RUNGS

echo "=== $(date -Is) ALL LEGS ATTEMPTED. Failures:"
grep -lE "FAIL|RED|Traceback" $OUT/*.log 2>/dev/null || echo "  none"
