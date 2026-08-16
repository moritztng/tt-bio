#!/bin/bash
# Blackhole neutrality on the assembled branch: state/japanfold-wh-cutover.md §4.2 and §4.3.
#
# Two legs on two cards of qb1, concurrent because they are independent single-card
# measurements. Card 1 runs the K3 A/B, which is the one change that reaches a 13x10 grid
# (§4.1 proved every other one inert). Card 0 runs the 13-leg perf gate.
#
# The A/B reuses perf/whb2/bh_ab.sh unchanged: a discarded warm-up, then A,B,A,B
# interleaved, then an A/A pair so the noise floor is measured in the same session as the
# delta rather than inherited from an earlier one. That protocol is what the boltz2
# campaign's three inconclusive rounds were missing, and qb1 is at loadavg 0.00 for the
# first time in the campaign.
set -u
TREE=${TREE:-/home/ttuser/.coworker/wt/japanfold-wh-cutover}
PY=${PY:-/home/ttuser/tt-bio/env/bin/python3}
OUT=$TREE/perf/whcut/out
mkdir -p "$OUT"
cd "$TREE" || exit 1
echo "CHAIN START $(date -u +%FT%TZ) head $(git rev-parse HEAD) load $(cut -d' ' -f1-3 /proc/loadavg)"

# §4.3, card 1. 640 because K3 fires there and the campaign measured it on Blackhole three
# times; 576 because it is K3's largest Wormhole win and has no Blackhole reading at all.
(
  for SIZE in 640 576; do
    PY=$PY TREE=$TREE CARD=1 SIZE=$SIZE REP=3 \
      OUT=$OUT/k3_bh_assembled_$SIZE \
      OFF="TT_BIO_SDPA_DIV_K=0" ON="TT_BIO_SDPA_DIV_K=1" \
      bash perf/whb2/bh_ab.sh
  done
  echo "K3 AB LEG DONE $(date -u +%FT%TZ)"
) > "$OUT/qb1_k3ab.log" 2>&1 &
AB=$!

# §4.2, card 0. The established 13-leg bar, same script the release gate runs on main.
(
  env TT_VISIBLE_DEVICES=0 TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$TREE" \
    TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
    "$PY" scripts/perf_regression.py
  echo "PERF GATE EXIT $?"
) > "$OUT/qb1_perf_regression.log" 2>&1 &
PG=$!

wait $AB; echo "ab leg rc=$?"
wait $PG; echo "perf leg rc=$?"
echo "CHAIN DONE $(date -u +%FT%TZ)"
