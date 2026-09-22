#!/usr/bin/env bash
# of3t-hostleg: everything downstream of the float64 input-embedder capture, in one chain.
#   the three arms -> the merged 17-tensor arm -> the model-denominator score -> the coverage
# Each step's output is an artifact in perf/of3t_hostleg/, so a failure stops the chain with the
# last good one on disk rather than half a table.
set -uo pipefail
# One at a time. Two launchers fired this chain once -- a detached watcher on qb1 and a session
# waiter that outlived its turn -- and the second was REWRITING device_grads_hl_allbreak.pt while
# the first was scoring it. Both runs were killed and the chain re-run; the lock is so it cannot
# happen again. flock, not a pidfile: the kernel releases it if the holder dies.
exec 9>/tmp/of3t-hostleg-finish.lock
if ! flock -n 9; then
  echo "REFUSING: another finish.sh holds /tmp/of3t-hostleg-finish.lock. Two of these write the"
  echo "same artifacts and want the same card; one of them would score a half-written dump."
  exit 3
fi
W=/home/ttuser/.coworker/wt/of3t-hostleg
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
D=/home/ttuser/of3t_hostleg
B=$D/ie_boundary.pt
export PYTHONPATH="$W/perf/of3t_tape:$W"
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:of3t-hostleg
export OMP_NUM_THREADS=4

echo "=== 1/5 the arm, the A/A and the BREAK on the float64 boundary  $(date -u +%FT%TZ) ==="
for spec in "_f64:" "_f64_aa:" "_f64_break:--break-cot"; do
  tag=${spec%%:*}; extra=${spec#*:}
  $PY perf/of3t_hostleg/ie_arm.py --boundary "$B" --tag "$tag" \
      --dump "$D/ie_grads${tag}.pt" $extra > "$D/ie_arm${tag}.log" 2>&1 || exit 1
  grep -E '"n_measurable"|"n_over_5e-2"|"worst_rel_l2"|"worst_tensor"|"break_cot"' "$D/ie_arm${tag}.log"
done
$PY -c "
import torch
a=torch.load('$D/ie_grads_f64.pt',map_location='cpu',weights_only=False)
b=torch.load('$D/ie_grads_f64_aa.pt',map_location='cpu',weights_only=False)
print('A/A:', sum(1 for k in a if torch.equal(a[k],b[k])), 'of', len(a), 'bit-identical')
" 2>&1 | grep -v DEBUG | grep -v 'Config{'

echo "=== 2/5 one arm out of the two boundaries  $(date -u +%FT%TZ) ==="
$PY perf/of3t_hostleg/merge_arms.py "$D/device_grads_hl_all.pt" \
    "$D/device_grads_hl_refatom.pt" "$D/ie_grads_f64.pt" || exit 1
$PY perf/of3t_hostleg/merge_arms.py "$D/device_grads_hl_allbreak.pt" \
    "$D/device_grads_hl_break.pt" "$D/ie_grads_f64_break.pt" || exit 1

echo "=== 3/5 the seventeen, in the model's denominator  $(date -u +%FT%TZ) ==="
OMP_NUM_THREADS=8 nice -n 10 $PY perf/of3t_hostleg/score_seventeen.py \
  --arm hostapplied=$D/device_grads_hl_shipped.pt \
  --arm all=$D/device_grads_hl_all.pt \
  --arm break=$D/device_grads_hl_allbreak.pt || exit 1

echo "=== 4/5 the coverage, re-measured  $(date -u +%FT%TZ) ==="
perf/of3t_hostleg/coverage.sh all || exit 1

echo "=== 5/5 done  $(date -u +%FT%TZ) ==="
echo FINISH_CHAIN_DONE
