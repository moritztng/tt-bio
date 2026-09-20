#!/usr/bin/env bash
# The D10 fold campaign: two arms x nine seeds, ARMS INTERLEAVED on one card.
#
# Interleaved because a card that drifts (clock, co-tenant, thermals) over the hour a campaign
# takes would otherwise put all of one arm on the drifted half. Two arms only, not four: a
# selection-rule change does not move the diffusion samples, so {shipped, D10} share one set of
# 45 structures and {D1, D1+D10} share the other, and the other two arms are produced by
# re-ranking the SAME samples in rules.py.
#
#   bash perf/of3t_confhead/campaign.sh <card> [seeds...]
set -u
CARD="${1:-0}"; shift || true
SEEDS=("${@:-}"); [ -z "${SEEDS[*]}" ] && SEEDS=(1 2 3 4 5 6 7 8 9)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-$HOME/tt-bio-dev/env/bin/python}"
OUT="${OUT:-$HOME/of3t_confhead_out/fold}"
cd "$ROOT" || exit 1
for s in "${SEEDS[@]}"; do
  for arm in fix ship; do
    if [ -f "$OUT/${arm}_s${s}/samples.json" ]; then
      echo "SKIP ${arm} s${s} (already has samples.json)"; continue
    fi
    # Retry, because a co-tenant taking the card is transient and silently losing a fold to
    # it is not: `fix s3` was dropped exactly that way when of3t-rebase opened card 0 for the
    # two minutes tt_bio's device lease waits before refusing. `${PIPESTATUS[0]}` and not `$?`,
    # which after a pipeline reports grep's status and so read rc=0 on that very failure.
    for try in 1 2 3; do
      echo "=== START ${arm} s${s} try${try} $(date -u +%H:%M:%S)"
      timeout 3600 "$PY" perf/of3t_confhead/rank_fold.py --arm "$arm" --seed "$s" \
        --card "$CARD" --msa-dir "$HOME/of3t_confhead_msa" --out-root "$OUT" 2>&1 \
        | grep -E "^\[|^  sample|^wrote|fold failed|DeviceInUse|Error|Traceback"
      rc=${PIPESTATUS[0]}
      echo "=== END   ${arm} s${s} try${try} rc=${rc} $(date -u +%H:%M:%S)"
      [ "$rc" -eq 0 ] && break
      [ "$try" -lt 3 ] && sleep 180
    done
  done
done
echo "CAMPAIGN_DONE $(date -u +%H:%M:%S)"
