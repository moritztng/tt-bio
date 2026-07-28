#!/usr/bin/env bash
# abag_xm_endgame.sh -- run the post-generation sequence in the one order that is correct.
#
# The steps and their ordering constraints are scattered across ~20 state-doc entries, and two of the
# constraints are easy to get wrong in ways that cost hours or silently ship half a dataset:
#
#   * LEARNED RANKERS BEFORE THE MERGE. DeepRank-Ab is ~82 s/fold; scoring after merging means one
#     host does all 492 (~11 h) instead of each host doing its own half (~5.6 h). The merge carries
#     ranker_scores.csv rows (8650e335), so a host's scoring survives it -- but only if it happened
#     first.
#   * DeepRank-Ab CANNOT RUN WHILE THE CARDS FOLD. It takes every core and ignores every cap, so it
#     is gated on the cards being idle. That is a scheduling constraint, not a bug.
#
# Stops before uploading. Publishing is Moritz's gate and this script never crosses it.
#
#   Usage:  scripts/abag_xm_endgame.sh <peer-host> [--force-incomplete]
#
# Idempotent: every step skips work already done, so re-running after a failure resumes.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PEER="${1:?usage: abag_xm_endgame.sh <peer-host> [--force-incomplete]}"
FORCE="${2:-}"
PY=/home/ttuser/.abag_xm_label_venv/bin/python3
[ -x "$PY" ] || PY=/home/ttuser/tt-bio/env/bin/python3
TIER=$HOME/abag_xm/tier_a
LOGDIR=$HOME/abag_xm/logs
mkdir -p "$LOGDIR"
step(){ echo; echo "=== $* ==="; }
die(){ echo "!! $*" >&2; exit 1; }

step "0. preconditions"
"$PY" - <<'PY' || die "could not read progress"
import json
from pathlib import Path
p = Path.home()/"abag_xm"/"tier_a"/"progress.jsonl"
recs=[json.loads(l) for l in open(p) if l.strip()]
ok={(r["target"],r["model"]) for r in recs if r.get("status")=="ok"}
print(f"  local ok folds: {len(ok)}")
PY
nlab=$(ls "$TIER"/labels/*.json 2>/dev/null | wc -l)
echo "  local labels: $nlab"
if pgrep -f "tt_bio.mai[n] predict" >/dev/null 2>&1; then
  echo "  cards are STILL FOLDING."
  if [ "$FORCE" != "--force-incomplete" ]; then
    die "DeepRank-Ab cannot be contained and would starve the folds. Wait for generation, or pass
    --force-incomplete to run everything except the learned rankers."
  fi
  SKIP_LEARNED=1
else
  SKIP_LEARNED=0
  echo "  cards idle -- learned rankers can run"
fi

step "0b. harness selftests"
# These existed and nothing ran them, which is how the ranker table shipped a positional join for
# weeks: `all_runs[k]` was paired with `samples[k]`, and since labels.py sorts model files by
# FILENAME (0, 1, 10, 11, ..., 2, ...) that put another structure's DockQ on 48 of every 50 rows.
# It is invisible downstream -- a within-target permutation leaves global Spearman at 0.79 while
# the per-target median collapses -- so it has to be caught here, before steps 2, 6 and 7 compute
# numbers on top of it. Hard failure: every quantity below inherits the join.
for t in abag_xm_ranker_join_selftest.py \
         abag_xm_merge_hosts_selftest.py \
         abag_xm_merge_ranker_selftest.py; do
  printf '  %-40s ' "$t"
  if out=$(PYTHONPATH="$WT" "$PY" "$WT/scripts/$t" 2>&1); then
    echo PASS
  else
    echo FAIL
    echo "$out" | tail -10
    die "$t failed -- fix it before assembling anything; every number below inherits this"
  fi
done

step "1. gaps across BOTH hosts (a per-host count is misleading)"
"$PY" "$WT/scripts/abag_xm_union_gaps.py" || echo "  (union check unavailable; continuing)"

step "2. learned rankers on THIS host, before the merge"
if [ "$SKIP_LEARNED" = "1" ]; then
  echo "  SKIPPED (cards busy). Re-run this script once they are idle; the merge in step 4 would"
  echo "  otherwise carry no ranker rows from this host."
else
  "$PY" "$WT/scripts/abag_xm_ranker_scores.py" --all --out "$TIER/ranker_scores.csv" \
      --with_deeprank --with_abagrank 2>&1 | tee -a "$LOGDIR/endgame_rankers.log" | tail -20
  echo "  NOTE: run this same step on $PEER before continuing, or its half is scored on this host"
  echo "        afterwards at twice the cost."
fi

step "3. confirm the peer has scored its own half"
if ssh -o BatchMode=yes "ttuser@$PEER" "test -s abag_xm/tier_a/ranker_scores.csv" 2>/dev/null; then
  n=$(ssh -o BatchMode=yes "ttuser@$PEER" "wc -l < abag_xm/tier_a/ranker_scores.csv" 2>/dev/null)
  echo "  $PEER has ranker_scores.csv ($n lines) -- the merge will carry it"
else
  echo "  !! $PEER has NO ranker_scores.csv. Merging now discards nothing, but its folds will have to"
  echo "     be scored here afterwards, at ~2x the wall. Run step 2 on $PEER first if you can."
fi

step "4. merge the peer in (coordinates, PAEs, labels, progress, ranker rows)"
"$PY" "$WT/scripts/abag_xm_merge_hosts.py" --peer "$PEER" --dry-run | tail -12
# Interactive confirmation when there is a terminal; otherwise require ENDGAME_MERGE=yes, so running
# this from a non-interactive context cannot merge by reading EOF as consent.
if [ -t 0 ]; then
  read -r -p "  proceed with the real merge? [y/N] " a
else
  a="${ENDGAME_MERGE:-}"
  echo "  non-interactive: ENDGAME_MERGE=${a:-<unset>}"
  [ "$a" = "yes" ] && a=y
fi
[ "$a" = "y" ] || die "stopped before merging (set ENDGAME_MERGE=yes to merge non-interactively)"
"$PY" "$WT/scripts/abag_xm_merge_hosts.py" --peer "$PEER" 2>&1 | tail -14

step "5. label anything the merge brought in unlabelled"
"$PY" "$WT/scripts/abag_xm_label_cost_model.py" || true
echo "  the labels loop picks these up automatically; this is the cost, not a command"

step "6. Phase 5 signal on the merged table"
"$PY" "$WT/scripts/abag_xm_phase5_signal.py" "$TIER/ranker_scores.csv" 2>&1 | tail -40

step "7. release tables + preflight, NO upload"
# There is no --dry-run: abag_xm_publish.py only assembles and checks unless given --go, and --go
# requires Moritz's explicit approval. Omitting it IS the safety mechanism, so it is omitted here
# and this script never passes it.
"$PY" "$WT/scripts/abag_xm_publish.py" --out_dir /tmp/abag_xm_release 2>&1 | tail -30

echo
echo "=== stopping here ==="
echo "Publishing to HuggingFace is Moritz's gate and this script does not cross it."
echo "Review /tmp/abag_xm_release, then publish deliberately."
