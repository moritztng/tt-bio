#!/usr/bin/env bash
# Size-ladder A/B for the token bucket, on qb2 (p300c).
#
# The standing arm cannot score here: docs/size_ladder_baseline.json holds p150a rows only, and
# p150a is qb1, admin-offline until 2026-08-31, so both ladder legs read "FAIL NO BASELINE for card
# type 'p300c'" and never folded. Recording a p300c baseline off this branch would also bake the
# flip in as the reference, which is the one thing a lever's own branch must not do.
#
# So: record the ladder with the bucket OFF into a scratch baseline outside the repo, then run the
# ON arm against it. Same instrument, same rungs, same reps, and the comparison is ON-vs-OFF on one
# card instead of ON-vs-a-committed-number-from-another-card.
#
# opendde only. protenix-v2's four rungs are all multiples of 64, so its pad is 0 at every one and
# its rows are provably inert (the 512 aa fold is byte-identical, cif 5e404779d791fa8f both arms).
# opendde's refiner axis is Ns = 2*n_res - n_GLY, unaligned at every rung, so this model is the
# whole informative content of the arm.
set -u
WT=/home/ttuser/.coworker/wt/tokenbucket-rebase-and-land
# v0.7.0 raised the declared pins (transformers>=5.5.0); tt-bio-dev/env is on 4.57.6.
PY=${GATE_PYTHON:-/home/ttuser/.coworker/rel070/relvenv/bin/python3}
CARD=${CARD:?}
OUT=$WT/perf/tokenbucket/ladderab
mkdir -p "$OUT"
cd "$WT" || exit 1
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:tokenbucket-rebase-and-land"

log () { echo "$(date -Is) $* load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/progress.log"; }

log "BEGIN off-arm record (card $CARD)"
env $LEASE PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm PATH=/home/ttuser/tt-bio/env/bin:$PATH \
    TT_BIO_PROTENIX_TOKEN_BUCKET=0 \
    "$PY" -u scripts/release_gate.py --model size-ladder --size-ladder-models opendde \
    --size-ladder-record --size-ladder-baseline /tmp/slb_off_p300c.json \
    > "$OUT/off_record.log" 2>&1
log "END off-arm record rc=$?"
cp -f /tmp/slb_off_p300c.json "$OUT/baseline_off_p300c.json" 2>/dev/null

# The recorder leaves every dark lever's reason as a TODO and the comparison FAILS on any dark
# lever that still has one. That guard is right for a committed baseline and wrong here: this file
# is a scratch OFF-arm control, so darkness is the reference state, not a finding. Left in, it
# buried ~15 real ON-vs-OFF deltas under 56 lines about the OFF arm.
"$PY" perf/tokenbucket/stamp_off_reasons.py /tmp/slb_off_p300c.json
cp -f /tmp/slb_off_p300c.json "$OUT/baseline_off_p300c.json"

log "BEGIN on-arm compare vs the off arm"
env $LEASE PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm PATH=/home/ttuser/tt-bio/env/bin:$PATH \
    TT_BIO_PROTENIX_TOKEN_BUCKET=1 \
    "$PY" -u scripts/release_gate.py --model size-ladder --size-ladder-models opendde \
    --size-ladder-baseline /tmp/slb_off_p300c.json \
    > "$OUT/on_vs_off.log" 2>&1
log "END on-arm compare rc=$?"

# The committed baseline is never touched; assert it, do not assume it.
git -C "$WT" diff --quiet docs/size_ladder_baseline.json \
  && log "committed size_ladder_baseline.json unchanged" \
  || log "WARNING committed size_ladder_baseline.json MODIFIED"
log "ALL DONE"
