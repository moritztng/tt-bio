#!/bin/sh
# The whole ladder, one fold at a time on pc card 0, resume-safe across worker passes.
#
# Detached on purpose (setsid): a single rung is 20-40 min at 1024 tokens and the ladder is
# twelve of them, so no one worker pass can hold it. Rooted in THIS slug's worktree, never a
# parent's, because a concluded slug's worktree is torn down under a live job.
#
# Order is 1024 first, then downward. 1024 is the question Moritz asked; a rung below it only
# earns its compute if 1024 fails, and record.py's results.jsonl is what makes the skip safe.
set -u
B=/home/moritz/.coworker/wt/bh-1024-of3-openbind-rf3/perf/bh1024
cd "$B" || exit 1
LOG=$B/campaign.log

# The card may still be held by the worker pass that launched this campaign. Waiting is
# correct rather than failing: the pass's own fold is a rung we want, and starting on top of
# it would only lose the device race (tt-bio's lease refuses the open, exit 75) and burn the
# rung as a non-result.
wait_for_card() {
  waited=0
  # Two reads, because either alone lies: lsof can report a chip free that a pool still
  # owns (memory japanfold-pool-excluded-chip-reads-free-via-lsof), and a pgrep for our own
  # engine misses any other holder. Busy if EITHER says busy.
  while sudo -n lsof /dev/tenstorrent/0 >/dev/null 2>&1 \
        || pgrep -f 'tt_bio.main predict' >/dev/null 2>&1; do
    waited=$((waited + 30))
    [ "$waited" -gt 7200 ] && { echo "GIVEUP card held >2h" >> "$LOG"; return 1; }
    sleep 30
  done
  return 0
}

done_already() {   # model rung probe already recorded?
  [ -f "$B/results.jsonl" ] && grep -q "\"model\": \"$1\", \"rung\": \"$2\"" "$B/results.jsonl" \
    && grep "\"model\": \"$1\", \"rung\": \"$2\"" "$B/results.jsonl" | grep -q "\"probe\": \"$3\""
}

# model:recycling:sampling -- the platform's own resolved values (catalog.ENGINE_DEFAULTS).
# Not lowered to make a rung pass: rf3's recycling also sets how many MSA samples the trunk
# sees, so dropping it changes memory AND quality and would answer a different question.
for spec in openfold3:3:200 openbind:3:200 rf3:10:50; do
  m=${spec%%:*}; rest=${spec#*:}; rec=${rest%%:*}; samp=${rest#*:}
  for rung in tile_1024 tile_896 tile_768 cut_640; do
    if done_already "$m" "$rung" off; then
      echo "SKIP $m $rung (recorded)" >> "$LOG"; continue
    fi
    wait_for_card || exit 3
    echo "START $m $rung rec=$rec samp=$samp $(date -u +%FT%TZ)" >> "$LOG"
    sh "$B/run_rung.sh" "$m" "$rung" "$rec" "$samp" off >> "$LOG" 2>&1
    echo "END   $m $rung $(date -u +%FT%TZ)" >> "$LOG"
    # A rung that passed answers the model. Everything below it is proven by implication for
    # a capacity question, so stop descending and move to the next model.
    if grep "\"model\": \"$m\", \"rung\": \"$rung\"" "$B/results.jsonl" 2>/dev/null \
         | grep -q '"status": "ok"'; then
      echo "PASS $m at $rung -- lower rungs not needed" >> "$LOG"; break
    fi
  done
done
echo "CAMPAIGN DONE $(date -u +%FT%TZ)" >> "$LOG"
