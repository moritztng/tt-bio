#!/usr/bin/env bash
# Step G of the token-bucket flip. Ordered by information per device-hour, not by RELEASING.md's
# listing order:
#
#   1. parity        — the correctness gate, the long pole, and the only arm that can block landing
#                      (opendde-abag carries a global_dockq >= 0.50 floor). Every leg here is at an
#                      unaligned token count, so every leg moves. That is the fix working.
#   2. perf          — scoped to the three models that reach the bucketing. Verified by grep: only
#                      tt_bio/protenix.py and tt_bio/opendde.py call bucketed_width /
#                      bucketed_pairformer / _token_bucket, so the other 13 models perf_regression
#                      would sweep cannot be affected and measuring them is wasted device time.
#                      Its input is examples/trpcage.yaml at 20 aa, which the bucket pads 20 -> 64,
#                      so this is the most extreme relative pad anywhere in the gate.
#   3. sizeladder298 — the off-64 diagnostic rung. The standing ladder is 256/512/640/768, every one
#                      a multiple of 64, so it is structurally blind to this lever. 298 is the only
#                      rung that can see it at all. Census only: report it, never re-record off it.
#   4. sizeladder    — the standing arm, last. Expected UNCHANGED, because no default rung is
#                      unaligned. Green here without re-recording is the blindness above, not a pass.
#
#     CARD=<your launch grant> bash perf/tokenbucket/gate_tokenbucket.sh
set -u
: "${CARD:?set CARD to this launch grant}"
WT=/home/ttuser/.coworker/wt/protenix-opendde-token-bucket-flip-measure
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/w6_dockq_py
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:protenix-opendde-token-bucket-flip-measure"
OUT=perf/tokenbucket/gate
mkdir -p $OUT

if ! timeout 300 env $LEASE PYTHONPATH=$WT "$PY" -u perf/tokenbucket/preflight_card.py \
     >/dev/null 2>&1; then
  echo "PREFLIGHT FAILED on card $CARD. Not gating."; exit 1
fi
echo "$(date -Is) preflight OK on card $CARD, load $(cut -d' ' -f1-3 /proc/loadavg)"

run() {  # rc is the command's, not an echo's (pass 2 logged rc=0 over four hard failures)
  tag=$1; shift
  if [ -f "$OUT/$tag.done" ]; then echo "SKIP $tag (already done)"; return 0; fi
  echo "=== $(date -Is) BEGIN $tag"
  # keep the previous attempt: a red leg has no .done, so it re-runs on every relaunch and
  # would otherwise overwrite the very output that documents why it is red.
  [ -f "$OUT/$tag.log" ] && mv "$OUT/$tag.log" "$OUT/$tag.$(date +%H%M%S).log"
  env $LEASE PYTHONPATH=$WT ESM_ROOT=$ESM_ROOT OPENDDE_DOCKQ_PYTHON=$OPENDDE_DOCKQ_PYTHON "$@" > "$OUT/$tag.log" 2>&1
  rc=$?
  echo "=== $(date -Is) END $tag rc=$rc"
  [ $rc -eq 0 ] && touch "$OUT/$tag.done"
  return 0   # never abort the chain on one red leg; every leg's log is wanted
}

# 1. Correctness. Own --workdir: full_parity_gate keys cached per-leg reports on leg id alone, so
# the shared /tmp/full_parity_gate would replay another tree's verdicts as this branch's.
run parity "$PY" -u scripts/full_parity_gate.py --workdir /tmp/full_parity_gate-tokenbucket \
  --workers qb1:$CARD \
  --leg protenix-prot-msa --leg protenix-ubq-msa --leg protenix-hsa-msa --leg protenix-9ncy-msa \
  --leg opendde-trpcage-nomsa --leg opendde-prot-prod --leg opendde-abag --leg capacity

# 2. Perf, only where the flip can reach.
for m in protenix-v2 opendde opendde-abag; do
  run "perf-$m" "$PY" -u scripts/perf_regression.py --model $m
done

# 3/4. The ladder: diagnostic rung first, then the standing arm.
export RELEASE_GATE_SIZE_RUNGS=298
run sizeladder298 "$PY" -u scripts/release_gate.py --model size-ladder \
  --size-ladder-models protenix-v2,opendde
unset RELEASE_GATE_SIZE_RUNGS
run sizeladder "$PY" -u scripts/release_gate.py --model size-ladder

echo "=== $(date -Is) ALL LEGS ATTEMPTED"
for f in $OUT/*.log; do
  printf '%-22s %s\n' "$(basename $f .log)" \
    "$([ -f "${f%.log}.done" ] && echo 'rc=0' || echo 'RED or not finished')"
done
