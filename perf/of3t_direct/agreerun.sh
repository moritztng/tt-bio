#!/usr/bin/env bash
# The direct comparison, per scope: our device gradient against upstream's own bf16 training
# step, with the shared float64 subtrahend removed. One scope per invocation.
#  $1 scope  $2 device.pt  $3 device-permuted.pt|-  $4 capture|-  $5 cap key  $6 cap prefix
set -euo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$W"
PIN=/home/ttuser/of3t_refprec/pinned_p175
SCOPE=$1; DEV=$2; DEVP=$3; CAP=$4; CKEY=${5:-grad_f64}; PFX=${6:-}
ARGS=(--device "$DEV"
      --f64 /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt
      --bf16 "$PIN/arm4_bf16_autocast/grads_f64.pt"
      --f32  "$PIN/arm2_f32_upstream/grads_f64.pt"
      --upstream-permuted /home/ttuser/of3t_trajectory/ref/negctl_permuted_grads.pt
      --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json
      --out "perf/of3t_direct/DIRECT_${SCOPE}.json"
      --sidecar-dir "perf/of3t_direct/sidecar_${SCOPE}"
      --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb
      --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548)
[ "$DEVP" != "-" ] && ARGS+=(--device-permuted "$DEVP")
[ "$CAP"  != "-" ] && ARGS+=(--diffcap "$CAP" --cap-key "$CKEY" --cap-prefix "$PFX")
OMP_NUM_THREADS=2 nice -n 15 /home/ttuser/tt-bio-dev/env/bin/python \
  perf/of3t_trajectory/agreement.py "${ARGS[@]}"
