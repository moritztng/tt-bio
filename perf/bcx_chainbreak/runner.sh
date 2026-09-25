#!/bin/bash
cd /home/ttuser/.coworker/wt/bcx-chainbreak/perf/bcx_chainbreak || exit 1
export PYTHONPATH=/home/ttuser/bcx_e2e/bc2 BC2_PARAMS=/home/ttuser/bcx_e2e/af2_params
PY=/home/ttuser/bcx_e2e_venv/bin/python
leg () { # leg binder target start chains seed
  tag="$1_$2_$3_$4_c$5_s$6"
  [ -f "out/$tag.json" ] && return
  $PY chainbreak.py run --leg "$1" --binder "$2" --target "$3" --target-start "$4" \
      --chains "$5" --seed "$6" --out "out/$tag.json" >/dev/null 2>"out/$tag.err"
  echo "$(date -u +%H:%M:%S) $tag rc=$?" >> out/runner.log
}
# H1: one chain, same 128 residues, both entry points
leg predict  64 64 1 1 0
leg gradient 64 64 1 1 0
# distributions: three more binder draws at each length, both entry points
for s in 1 2 3; do
  for l in predict predict_identity; do
    leg $l 32 96 1 2 $s
  done
done
for s in 1 2 3; do
  for l in predict predict_identity; do
    leg $l 32 160 1 2 $s
  done
done
for s in 1 2; do
  for l in predict predict_identity; do
    leg $l 32 288 1 2 $s
  done
done
# H2 negative control at the second length too
leg predict          32 160 4000 2 0
leg predict_identity 32 160 4000 2 0
echo "$(date -u +%H:%M:%S) RUNNER_DONE" >> out/runner.log
