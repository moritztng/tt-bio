#!/bin/bash
# One `hostcores.py` arm per XLA:CPU configuration, each in its own process because XLA_FLAGS
# is read once at backend init. Same box, same sitting, back to back, so the arms are
# comparable to each other; the absolute seconds carry whatever load the box had.
#   flagsweep.sh flags     the configuration ladder
#   flagsweep.sh cores     the affinity ladder, default flags
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/moritz/bcx_tail/venv/bin/python3}
OUT=${OUT:-perf/bcx_p10_hostcut/out}
mkdir -p "$OUT"
run() {  # run <tag> <xla_flags> [taskset cpu spec]
    local tag=$1 flags=$2 cpus=${3:-}
    local pre=()
    [ -n "$cpus" ] && pre=(taskset -c "$cpus")
    XLA_FLAGS="$flags" "${pre[@]}" "$PY" -u scripts/bcx_p10_hostcut/hostcores.py \
        --tag "$tag" --reps "${REPS:-8}" --out "$OUT/$tag.json" >/dev/null 2>"$OUT/$tag.log" \
        || { echo "$tag FAILED -- $OUT/$tag.log"; return 0; }
    "$PY" -c "
import json
d=json.load(open('$OUT/$tag.json'))
f,b=d['forward'],d['forward_and_backward']
print("%-26s fwd %6.3f s %4.2f cores | fwd+bwd %6.3f s %4.2f cores | load1 %5.2f | aff %2d'
      % (d['tag'], f['median_wall_s'], f['median_cores'],
         b['median_wall_s'], b['median_cores'], b['median_load1'], d['affinity_cores']))
"
}
ncpu=$(nproc)
case "${1:-flags}" in
flags)
  run base            ""
  run xnn_graph       "--xla_cpu_experimental_xnn_graph_fusion_mode=XNN_GRAPH_FUSION_MODE_GREEDY"
  run onednn          "--xla_cpu_use_onednn=true"
  run vec512          "--xla_cpu_prefer_vector_width=512"
  run fastruntime     "--xla_cpu_opt_preset=FAST_RUNTIME"
  run noxnn           "--xla_cpu_use_xnnpack=false"
  run noeigenthreads  "--xla_cpu_multi_thread_eigen=false"
  run fastmath        "--xla_cpu_enable_fast_math=true"
  ;;
cores)
  for c in 1 2 3 4 6 8 $ncpu; do
      run "cores$c" "" "0-$((c-1))"
  done
  ;;
esac
