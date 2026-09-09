#!/bin/bash
# The 1536 bar first, for every model in the registry. A model that clears it needs no ladder;
# a model that does not gets one walked downward in a later pass.
WT=/home/ttuser/.coworker/wt/bh-1536-structure
cd $WT || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
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
for spec in "$@"; do
  IFS=: read -r model size tag <<< "$spec"
  if [ -z "$tag" ] && $PY perf/bh1536/measured.py "$model" "$size"; then
    echo "skip $model $size (already measured)"; continue
  fi
  flock 9
  echo "=== $(date -u +%FT%TZ) $model $size ${tag:+tag=$tag} ==="
  $PY perf/bh1536/run_rung.py --model "$model" --size "$size" --budget 2700 \
      ${tag:+--tag "$tag"}
  flock -u 9
done
echo "CHAIN DONE $(date -u +%FT%TZ)"
