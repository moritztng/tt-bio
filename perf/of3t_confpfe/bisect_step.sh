#!/usr/bin/env bash
# of3t-confpfe: one `git bisect run` step, run inside the scratch worktree at the commit under test.
# 125 (skip) when the probe cannot run on this tree; otherwise bisect_quick.py's verdict.
W=/home/ttuser/.coworker/wt/of3t-confpfe; S=/home/ttuser/of3t_confpfe; T=$PWD
C=$(git rev-parse --short HEAD); O=$S/bisect/$C; mkdir -p $O
source /home/ttuser/tt-bio-dev/env/bin/activate
env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-confpfe PYTHONPATH=$T \
  timeout 400 python3 $W/perf/of3t_confpfe/head_probe.py --modes bf16_tape \
  --dump $S/probe_b/conf_boundary_none.pt --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
  --out-dir $O > $O/probe.log 2>&1 || { echo "$C skip (probe failed)" >> $S/bisect/steps.log; exit 125; }
PYTHONPATH=/home/ttuser/of3t-campaign-refs/of3pkg043:/home/ttuser/of3t_refprec/deps \
  python3 $W/perf/of3t_confpfe/bisect_quick.py /home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt \
  $O/conf_bf16_tape.pt > $O/check.log 2>&1
rc=$?
echo "$C rc=$rc $(tail -1 $O/check.log)" >> $S/bisect/steps.log
exit $rc
