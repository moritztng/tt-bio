#!/usr/bin/env bash
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=${OF3T_REBASE_RUN:-$HOME/of3t_rebase_run}
PY=/home/ttuser/tt-bio-dev/env/bin/python
# The two checkouts this compares are supplied, not assumed: of3t-rebase kept them at
# $R/wt and $R/main_wt and both went with the worktree. OURS defaults to the checkout this
# script lives in; MAIN has no default, because guessing one is how a comparison ends up
# reading the same tree twice.
OURS=${OURS:-$W}
MAIN=${MAIN:-}
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-rebase
export OMP_NUM_THREADS=4
mkdir -p "$R/run"
cd "$W"
[ -n "$MAIN" ] || { echo "set MAIN=<a checkout of main> -- see the comment above"; exit 2; }
for arm in "ours:$OURS:" "main:$MAIN:" "ours-perturbed:$OURS:--perturb"; do
  tag=${arm%%:*}; rest=${arm#*:}; repo=${rest%%:*}; flag=${rest#*:}
  echo "=== $tag  $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_rebase/identity.py --repo "$repo" --out "$R/run/identity_$tag.json" $flag
  echo "=== $tag exit $? ==="
done
echo "IDENT_ALLDONE $(date -u +%FT%TZ)"
