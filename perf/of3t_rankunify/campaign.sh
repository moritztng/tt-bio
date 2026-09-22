#!/usr/bin/env bash
# The rank-unification fold campaign: one fold per (model, target, seed), all four cards.
#
# A ranking rule is post-forward, so one fold prices every candidate rule by re-ranking its
# recorded scalars offline. That is why this covers four models instead of one.
#
#   bash perf/of3t_rankunify/campaign.sh <model> <card> <target> <gt> <seed>...
set -u
MODEL="$1"; CARD="$2"; TGT="$3"; GT="$4"; shift 4
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-$HOME/tt-bio-dev/env/bin/python}"
cd "$ROOT" || exit 1
for s in "$@"; do
  for try in 1 2; do
    echo "=== START $MODEL $(basename "$TGT" .yaml) s$s card$CARD try$try $(date -u +%H:%M:%S)"
    TT_BIO_LEASE_CARDS="$CARD" timeout 3600 "$PY" perf/of3t_rankunify/fold_one.py \
      --model "$MODEL" --target "$TGT" --gt "$GT" --seed "$s" --card "$CARD" 2>&1 \
      | grep -E "^OK|^FAIL|^SKIP|Error|Traceback|AssertionError"
    rc=${PIPESTATUS[0]}
    echo "=== END   $MODEL s$s rc=$rc $(date -u +%H:%M:%S)"
    [ "$rc" -eq 0 ] && break
    [ "$try" -lt 2 ] && sleep 30
  done
done
echo "CAMPAIGN_DONE $MODEL $(basename "$TGT" .yaml) card$CARD $(date -u +%H:%M:%S)"
