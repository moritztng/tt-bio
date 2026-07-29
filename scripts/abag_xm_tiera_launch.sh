#!/usr/bin/env bash
# Launch (or resume) Tier-A generation for this host's cards, deriving the target slices from the
# committed cost-balanced split. Use for the first launch AND for resume passes: generate.py's
# done_pairs() skips only status=ok, so re-running the identical command picks up exactly the
# outstanding (target, model) pairs and nothing else.
#
# Why this script exists: the slice->card assignment previously lived only in an ad-hoc file outside
# the repo, so the launch could not be reproduced from the checkout alone. This host's share is now
# derived deterministically from
# docs/implementation-parity-data/abag-xm-tier-a-slices.json (slices_8), which is
# longest-processing-time-balanced by token count (0.3%/2.8% imbalance vs 9.9%/15.7% for index
# slicing):
#     tt-quietbox2 -> slices 0..3        tt-quietbox -> slices 4..7
# The two hosts' shares are DISJOINT, which is what prevents cross-host double-folding, since
# progress.jsonl is host-local and cannot dedupe across machines.
#
# Card affinity is NOT meaningful: which card runs which slice is irrelevant because done_pairs()
# dedupes on (target, model), not on card. What IS essential is that every one of this host's
# slices reaches some card -- so when fewer cards are available than slices, the leftover slices
# are APPENDED to the launched cards rather than dropped. Dropping them would silently lose ~20
# targets per unavailable card, which is exactly the failure this note exists to prevent.
#
#   Usage:  scripts/abag_xm_tiera_launch.sh [cards] [slices]
#           cards  defaults to "0 1 2 3"; pass a subset when another worker holds a card.
#           slices defaults to this host's own four; pass a list to take over the other
#                  host's share, which is ONLY safe while that host is not running.
#
# --concurrent_folds is always 4, not the number of cards launched: sizing as if the box were full
# leaves headroom for a sibling worker instead of taking every core. It sets host_threads via
# cores//concurrent_folds, so qb1 gets 8 and qb2 gets 4 automatically.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/ttuser/tt-bio/env/bin/python3
SLICES="$WT/docs/implementation-parity-data/abag-xm-tier-a-slices.json"
LOGDIR=$HOME/abag_xm/logs
CARDS="${1:-0 1 2 3}"
# Optional explicit slice list, for taking over another host's share. The default is this
# host's own four, which is what keeps the two hosts disjoint; pass a list ONLY when the
# other host is not running, because progress.jsonl is host-local and cannot dedupe across
# machines. Added when qb2 hard-hung mid-campaign (2026-07-27) and the documented failover
# would otherwise have meant hand-editing this script under time pressure.
#   scripts/abag_xm_tiera_launch.sh "0 1 2 3" "0 1 2 3 4 5 6 7"   # qb1 takes everything
SLICE_LIST="${2:-}"
# Optional 3rd arg: comma-separated model list (default: generate.py's own default, the
# three MSA-fed generators). The esmfold2 leg launches as: <script> "<cards>" "" esmfold2
MODELS="${3:-}"
LEASE="worker:$(basename "$WT")"

case "$(hostname)" in
  tt-quietbox2) BASE=0 ;;
  tt-quietbox)  BASE=4 ;;
  *) echo "unknown host $(hostname): no slice assignment defined" >&2; exit 1 ;;
esac
if [ -n "$SLICE_LIST" ]; then
  MY_SLICES="$SLICE_LIST"
  echo "!! slice override: this host takes slices [$MY_SLICES] -- only valid while the other host is idle"
else
  MY_SLICES="$((BASE + 0)) $((BASE + 1)) $((BASE + 2)) $((BASE + 3))"
fi

mkdir -p "$LOGDIR" "$HOME/abag_xm/tier_a"
[ -f "$SLICES" ] || { echo "missing $SLICES" >&2; exit 1; }

# Recover folds that finished after their driver died before relaunching. A driver writes the
# progress.jsonl record when a fold returns, so if the driver dies mid-fold the fold keeps running
# (own session), completes, writes its CIFs + results.json, and is never recorded -- done_pairs()
# then hands it to a card a second time. That happened to 8 folds across both hosts on 2026-07-28
# and cost ~3.1 card-hours of recomputation to notice by hand. Reconciling here makes every resume
# self-healing: it only ever appends records derived from artifacts already on disk (flagged
# `recovered: true`), and it is a no-op when there is nothing orphaned.
"$PY" "$WT/scripts/abag_xm_reconcile_orphans.py" --write 2>&1 | tail -1

ncards=$(echo "$CARDS" | wc -w)
# Deal this host's 4 slices round-robin onto the available cards, so a card subset redistributes
# the work instead of dropping slices.
i=0
declare -A ASSIGN=()
for s in $MY_SLICES; do
  card=$(echo "$CARDS" | cut -d' ' -f$(( i % ncards + 1 )))
  ASSIGN[$card]="${ASSIGN[$card]:-}${ASSIGN[$card]:+,}${s}"
  i=$((i + 1))
done

for card in $CARDS; do
  slices="${ASSIGN[$card]:-}"
  [ -n "$slices" ] || { echo "card $card: no slices assigned, skipping"; continue; }
  targets=$("$PY" - "$SLICES" "$slices" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["slices_8"]
out = []
for s in sys.argv[2].split(","):
    out += d[s]
print(",".join(out))
PY
)
  [ -n "$targets" ] || { echo "card $card: empty target list for slices $slices" >&2; continue; }
  setsid nohup env TT_VISIBLE_DEVICES="$card" TT_BIO_LEASE_HOLDER="$LEASE" \
    PYTHONPATH="$WT" PYTHONUNBUFFERED=1 \
    "$PY" -u "$WT/scripts/abag_xm_generate.py" \
      --device "$card" --concurrent_folds 4 ${MODELS:+--models "$MODELS"} --targets "$targets" \
    > "$LOGDIR/gen_card$card.log" 2>&1 < /dev/null &
  echo "card $card <- slices $slices ($(echo "$targets" | tr ',' '\n' | wc -l) targets)"
done

sleep 20
echo "--- banner check: host_threads must be non-empty, mps must be 5 ---"
for card in $CARDS; do
  printf 'card %s: ' "$card"
  head -1 "$LOGDIR/gen_card$card.log" 2>/dev/null \
    | grep -oE "device=[0-9]+|targets=[0-9]+|mps=[0-9]+|host_threads=[0-9]+" | tr '\n' ' '
  echo
done
echo "--- coverage: all 4 of this host's slices must appear above ---"
