#!/usr/bin/env bash
set -uo pipefail
R=/home/ttuser/of3t_rebase
PY=/home/ttuser/tt-bio-dev/env/bin/python
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-rebase
export OMP_NUM_THREADS=4
cd "$R"
for arm in "ours:$R/wt:" "main:$R/main_wt:" "ours-perturbed:$R/wt:--perturb"; do
  tag=${arm%%:*}; rest=${arm#*:}; repo=${rest%%:*}; flag=${rest#*:}
  echo "=== $tag  $(date -u +%FT%TZ) ==="
  "$PY" identity.py --repo "$repo" --out "run/identity_$tag.json" $flag
  echo "=== $tag exit $? ==="
done
echo "IDENT_ALLDONE $(date -u +%FT%TZ)"
