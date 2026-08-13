#!/bin/bash
# Session 5 (qb1): the 1024 ratio, one arm per process, because in-process is now proven impossible
# on BOTH hosts.
#
# What changed. Section 10.2 and 11.2 concluded the second-fold stall was a qb2-stack fact and that
# qb1 was immune. That conclusion was UNDER-POWERED and is now refuted by this pass: at 1024 aa on
# qb1 the interleaved run completed folds 1 and 2 and then STALLED ON FOLD 3, 888 s against a
# 282.43 s warm fold (3.14x, past the pre-registered 3x threshold), with the py-spy stack IDENTICAL
# to qb2's:
#     timed_call (fold_ab512.py:94)   <- ttnn.synchronize_device
#     update_msa (tt_bio/protenix.py:2107)
#     _msa (tt_bio/protenix.py:2174) -> _trunk_cond (tt_bio/protenix.py:1874)
# 10.2 only ran TWO folds at 768 and so could not see a fold-3 onset. The stall is a protenix defect
# that crosses board and ttnn version; only its ONSET differs (qb2 trips at fold 2 at both 768 and
# 1024; qb1 survives four folds at 768 and trips at fold 3 at 1024).
#
# So 2.2 was right that 1024 is "not obtainable in-process", and it is right for qb1 too. The ratio
# has to come from one arm per process. That is cheap now in a way it was not before: 11.3 measured
# the per-process cold cost at 7.55 s at 768 once the ttnn kernel cache is warm, and this run's own
# 1024 kernels were compiled into that cache minutes ago, so each process should land near the
# 282.43 s warm fold rather than near the 363.42 s first-compile one.
#
# Three processes, not two: on, on, base4. The two `on` processes give a CROSS-PROCESS A/A floor,
# which is the only kind available here and is strictly worse than the in-process 0.550 s at 768,
# so it must be measured rather than assumed.
#
# REGISTERED PREDICTIONS (committed before the run; do not edit after it starts):
#   S1  Each process's fold lands 282-305 s, i.e. near the warm 282.43 s and well under the 363.42 s
#       first-compile fold, because the kernel cache is warm. If a process lands near 363 s the
#       cache did not carry over and S2's ratio is confounded by unequal compile costs.
#   S2  Cross-process A/A floor |on_p1 - on_p2| < 8 s. The in-process floor at 768 was 0.550 s;
#       across processes the per-process cold cost varies too, so this is deliberately loose.
#   S3  base4/on at 1024 < 1.0540x, the 768 value, continuing the monotone decay of the curve.
#       Predict 1.01-1.05x. E6 and K2 serve 0 at 1024 (11.4), and base4 has them off as well, so
#       as at 768 this ratio is what K1 + K1b alone buy.
#   Stop rule: if the A/A floor exceeds one third of the (base4 - on) gap, the 1024 ratio is
#   reported NOT RESOLVED rather than as a number.
#   Stop rule: any process that does not return is killed by explicit pid and its arm is dropped;
#   a partial set is reported as partial.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-sizes-perf
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-v2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
O=perf/pxsizes

echo "=== host: $(hostname)  card: $TT_VISIBLE_DEVICES  start: $(date -Is) ==="
sha256sum $WT/perf/size512/fixtures/cdk2x2_1024.yaml $WT/perf/size512/fixtures/cdk2x2_1024.a3m

one () {  # one <arm> <out>
  echo "=== $(date -Is)  1024 arm $1 (one arm, one process) ==="
  timeout 900 $PY -u perf/size512/fold_ab512.py --sizes 1024 --arms "$1" --skip-cold \
      --timers full --out "$2"
  echo "RC_1024_$1=$?"
}

$BL protenix-v2-sizes-perf -- bash -c '
  set -u
  cd '"$WT"'
  '"$(declare -f one)"'
  PY='"$PY"'; O='"$O"'
  one on    $O/s5_1024_on_p1.json
  one on    $O/s5_1024_on_p2.json
  one base4 $O/s5_1024_base4.json
'
echo "=== done: $(date -Is) ==="
