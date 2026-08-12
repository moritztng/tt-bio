#!/usr/bin/env bash
# One benchlocked session, arms alternating: on B on B.
#
#   on = a pristine detached worktree at 6a8d4f59 (origin/main, the branch base)
#   B  = this worktree, with the rollout hoists
#
# Both arms run the SAME harness (perf/of3deep/decomp.py, copied into the base worktree as
# an untracked file) so every row is comparable and the region table shows which rows moved.
# Two runs per arm give an A/A floor per arm and bracket the session.
set -eu
PY=/home/ttuser/tt-bio-dev/env/bin/python3
WT=/home/ttuser/.coworker/wt/openfold3-512aa-deep-perf
BASE=/tmp/of3_base
OUT=$WT/perf/of3deep/ab
mkdir -p "$OUT"
for i in 1 2; do
  for arm in on B; do
    [ "$arm" = on ] && root=$BASE || root=$WT
    echo "=== arm $arm run $i  ($root) ==="
    (cd "$root" && TT_VISIBLE_DEVICES=3 \
      TT_BIO_LEASE_HOLDER=worker:openfold3-512aa-deep-perf \
      PYTHONPATH="$root" "$PY" perf/of3deep/decomp.py --out "$OUT/${arm}_$i.json") \
      2>&1 | grep -E "^  (cold|top:|diff:|dm:|host:|trunk:|prep:)|^=== run"
  done
done
