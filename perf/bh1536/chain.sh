#!/bin/bash
# The 1536 bar first, for every model in the registry. A model that clears it needs no ladder;
# a model that does not gets one walked downward in a later pass.
WT=/home/ttuser/.coworker/wt/bh-1536-structure
cd $WT || exit 1
# Overridable so the circuit breaker below can be tested against a stub that returns a
# chosen exit code, instead of needing a wedged card to reproduce.
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
export PATH=/home/ttuser/.tenstorrent-venv/bin:$PATH

# One rung on card 0 at a time, however many chains are queued. BLOCKING flock, never -n:
# `flock -n` exits silently when the lock is held and the caller reads that as "nothing to do"
# (fleet-sh-flock-silent-exit-hides-dispatch), which here would drop the rung instead of
# waiting for it. This replaces the pgrep-absence waits chain2.sh/chain3.sh used: those keyed
# on a chain script's ABSENCE, and the `bash -c ... setsid nohup ./chain2.sh &` launcher stays
# alive with the chain as its child, so its cmdline kept matching and chain3 waited forever
# (measured: 19 min, zero rungs). A lock does not care what the process tree looks like.
# flock is not fair, so a chain can be passed over; with rungs this long that costs order, not
# correctness, and every rung is idempotent and skipped once measured.
exec 9>"$WT/perf/bh1536/.card0.lock"
# Skip a rung only if it was actually MEASURED. A CONTENDED row means a co-tenant held card 0
# and nothing ran, so skipping on the row's mere existence would retire the rung on the one
# outcome that carries no information (cost: protenix-v1 at 1536, scored FAIL behind a
# sibling's unpinned pytest). Asking run_rung.py's own reader rather than grepping for a field
# order also survives a new key being added to the row.
# spec is model:size or model:size:tag. A tagged rung always runs: it is a deliberate re-walk
# (a code change since the first one, a --debug repeat) and must not be retired by the untagged
# row it exists to re-examine, nor retire that row itself -- measured.py only counts tag "".
# A wedged card fails every rung in about 20 s, so a chain that ploughs on empties its whole
# queue into a dead chip: chain2 lost eight rungs that way in under five minutes -- the entire
# esmfold2 ladder and every 1300-token rung -- and then exited with nothing left to re-walk.
# None of it was published as a ceiling (they all read WEDGED, which measured.py does not count),
# but the queue was gone. Stop after three in a row instead and let a relaunch pick the list up.
nomeasure=0
for spec in "$@"; do
  IFS=: read -r model size tag <<< "$spec"
  if [ -z "$tag" ] && $PY perf/bh1536/measured.py "$model" "$size"; then
    echo "skip $model $size (already measured)"; continue
  fi
  flock 9
  # The lock serialises the chains that take it. A chain launched BEFORE the lock existed is
  # still walking its own list from a loop parsed in memory and cannot be made to take it, so
  # also wait on the thing that actually opens the card. Keyed on run_rung.py, the card user
  # itself, not on a chain wrapper's cmdline -- that is the mistake chain3.sh sat in for 19
  # minutes. Inside the lock, any run_rung.py seen here belongs to a lock-blind chain.
  while pgrep -f "bh1536/run_rung\.py" > /dev/null; do sleep 20; done
  echo "=== $(date -u +%FT%TZ) $model $size ${tag:+tag=$tag} ==="
  # A tagged rung is a deliberate re-walk of something already recorded, so it runs with
  # --debug: that keeps the worker's stdout connected and is the only way an engine line
  # (the pair-FFN fallback saying it fired) reaches fold.log and then the row.
  $PY perf/bh1536/run_rung.py --model "$model" --size "$size" --budget 2700 \
      ${tag:+--tag "$tag" --debug}
  rc=$?
  flock -u 9
  if [ "$rc" -eq 75 ]; then
    nomeasure=$((nomeasure + 1))
    if [ "$nomeasure" -ge 3 ]; then
      echo "STOPPING: 3 rungs in a row measured nothing (card contended or wedged). The rest of" \
           "the queue is left un-walked on purpose; reset the card and relaunch this chain."
      exit 75
    fi
  else
    nomeasure=0
  fi
done
echo "CHAIN DONE $(date -u +%FT%TZ)"
