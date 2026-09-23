#!/usr/bin/env bash
# of3t-covadopt: the model-boundary instrument over refcov's composed 3,660.
#   score.sh <run-tag>
# CPU only, no device is opened. Every arm it reads is already on disk.
set -uo pipefail
W=${W:-/tmp/of3t/of3t-covadopt/wt}
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=/tmp/of3t/of3t-covadopt
TAG=${1:-A}
mkdir -p "$O"

OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py \
  --arm "renorm:diffusion=$O/device_grads_rc_refatom_on.pt" \
  --arm "renorm:input_embedder=$O/ie_grads_f64_renorm_on.pt" \
  --arm "renorm:cond=/home/ttuser/of3t_wholemodel/cond_grads_renorm.pt" \
  --arm "renorm:aux=/home/ttuser/of3t_wholemodel/aux_grads_renorm.pt" \
  --arm "renorm:msa=/home/ttuser/of3t_wholemodel/msa_grads_renorm.pt" \
  --arm "renorm:pairformer_stack=/home/ttuser/of3t_trunkceiling/dev_RENORM_n384_nocaptures.pt" \
  --unwrap-key grads \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 /home/ttuser/of3t_refprec/pinned_p175/arm4_bf16_autocast/grads_f64.pt \
  --f32  /home/ttuser/of3t_refprec/pinned_p175/arm2_f32_upstream/grads_f64.pt \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect float64=1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --out "$O/MODEL_withtrunk_composed3660_n384.$TAG.json" \
  --sidecar-dir "$O/sidecar_composed3660.$TAG"
