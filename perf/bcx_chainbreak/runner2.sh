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
case "$1" in
  a) for s in 2 3; do for l in predict predict_identity; do leg $l 32 96 1 2 $s; done; done
     leg predict 32 160 4000 2 0; leg predict_identity 32 160 4000 2 0 ;;
  b) for s in 1 2 3; do for l in predict predict_identity; do leg $l 32 160 1 2 $s; done; done ;;
  c) for s in 1 2; do for l in predict predict_identity; do leg $l 32 288 1 2 $s; done; done ;;
esac
echo "$(date -u +%H:%M:%S) RUNNER_$1_DONE" >> out/runner.log
