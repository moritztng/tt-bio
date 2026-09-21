#!/bin/sh
# Reproduces every D107 number in `~/.coworker/state/of3t-d10-d107.md`. CPU only, no card.
#
# One interpreter, not two: `upstream_arm.py` needs `pytorch_lightning` and `torchmetrics`
# only as imports -- `grad_manager.py` names `pl` in an import and a QUOTED annotation
# (:18, :106) and builds a metric only under `log_grad_norm`, which is False here. `plstub/`
# supplies both as modules that RAISE if anything touches them, so the arm stays upstream's
# code and the claim that the stubs are inert is enforced rather than asserted.
set -e
WT=${WT:-/home/ttuser/.coworker/wt/of3t-d10-d107}
S=${S:-/home/ttuser/of3t_d10d107}          # scratch: the two trees and the outputs
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python}
REF=${REF:-/home/ttuser/of3t_refprec/of3pkg043}     # OpenFold3 0.4.3
PRE=${PRE:-2ed841c84}                               # wk/of3t before the D107 fix
POST=${POST:-cf83561d9}                             # of3t-optsem's fix, one file
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$S/out"
cd "$WT"
for r in "pre $PRE" "post $POST"; do
  set -- $r
  rm -rf "$S/$1"; mkdir -p "$S/$1"
  git archive "$2" tt_bio perf/of3t_optsem perf/of3t_rebind | tar -x -C "$S/$1"
done
cp -r "$HERE/plstub" "$S/plstub"
cp "$HERE/d107_score.py" "$HERE/show.py" "$S/"

cd "$S/post"
for sc in never single; do
  $PY perf/of3t_optsem/make_grads.py $sc "$S/out/grads_$sc.npz" > /dev/null
  PYTHONPATH=$REF:$S/plstub $PY perf/of3t_optsem/upstream_arm.py \
      "$S/out/grads_$sc.npz" "$S/out/up64_$sc.json" --dtype float64 > /dev/null
  PYTHONPATH=$REF:$S/plstub $PY perf/of3t_optsem/upstream_arm.py \
      "$S/out/grads_$sc.npz" "$S/out/up32_$sc.json" --dtype float32 > /dev/null
done
for a in pre post; do
  cd "$S/$a"
  for sc in never single; do
    $PY perf/of3t_optsem/ours_arm.py "$S/out/grads_$sc.npz" "$S/out/ours_${a}_$sc.json" > /dev/null
  done
done
cd "$S"
for sc in single never; do $PY d107_score.py out --scenario $sc > /dev/null; done
$PY show.py
