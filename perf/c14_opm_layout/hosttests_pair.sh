#!/usr/bin/env bash
# Host suite, branch vs an origin/main control, identical interpreter and identical invocation,
# run SEQUENTIALLY in detached /tmp trees so neither run perturbs the live gate worktree.
# TT_VISIBLE_DEVICES is empty on purpose: no card is opened by either side.
set -u
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/.coworker/wt/c14-stack-land/perf/c14_opm_layout
for pair in "ctl:/tmp/c14sl_ctl" "brn:/tmp/c14sl_brn"; do
  tag=${pair%%:*}; tree=${pair#*:}
  echo "=== $tag $(git -C "$tree" rev-parse HEAD) start $(date -u +%FT%TZ) ==="
  ( cd "$tree" && TT_VISIBLE_DEVICES= PYTHONPATH="$tree" ESM_ROOT=/home/ttuser/esm \
      "$PY" -m pytest tests -q --no-header -p no:cacheprovider ) \
      > "$OUT/hosttests_$tag.log" 2>&1
  echo "=== $tag rc=$? end $(date -u +%FT%TZ) ==="
done
