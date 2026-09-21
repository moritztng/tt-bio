#!/usr/bin/env bash
# PROTOCOL SS3e, with A14s sizing rule: the perturbation is chosen against the MEASURED
# baseline and the bar, not fixed at 1 %. Three arms on one boundary:
#   zero    every device gradient replaced by zeros -- the zero-model answer MEASURED
#   x1.01   the protocols nominal 1 %, which on this baseline does NOT cross the bar
#   x1.10   sized to cross it, on the tensor the sections mass lives in
set -uo pipefail
cd /home/ttuser/.coworker/wt/of3t-auxheads
T=aux_heads.distogram.linear.weight
for arm in zero x1.01 x1.10; do
  case $arm in
    zero)  A="--zero-model" ;;
    x1.01) A="--perturb $T:1.01" ;;
    x1.10) A="--perturb $T:1.10" ;;
  esac
  echo "=== arm $arm $(date -u +%FT%TZ) ==="
  bash perf/of3t_auxheads/auxrun.sh control_043_$arm $A
done
echo CONTROLS_ALLDONE $(date -u +%FT%TZ)
