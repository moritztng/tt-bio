#!/usr/bin/env bash
# of3t-ditmodel: assemble the four same-batch scopes into one model-scope reading, per tensor.
# CPU only. of3t-wholemodel's model_scope.py, unmodified; only the arms and the output path are
# this row's. Both arms are re-taken on TODAY's tree in the same hour, so the union is one tree:
#   d56on   the SHIPPED DEFAULT (nothing exported; autograd.SOFTMAX_BW_RENORM defaults True)
#   d56off  the BREAK CONTROL   (TT_BIO_SOFTMAX_BW_RENORM=0)
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
PIN=/home/ttuser/of3t_refprec/pinned_p175
D=/tmp/of3t/of3t-ditmodel
ARGS=()
for a in d56on d56off; do
  ARGS+=(--arm "$a:diffusion=$D/device_grads_043all_$a.pt")
  ARGS+=(--arm "$a:cond=$D/cond_grads_$a.pt")
  ARGS+=(--arm "$a:aux=$D/aux_grads_$a.pt")
  ARGS+=(--arm "$a:msa=$D/msa_grads_$a.pt")
done
OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py "${ARGS[@]}" \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 "$PIN/arm4_bf16_autocast/grads_f64.pt" \
  --f32  "$PIN/arm2_f32_upstream/grads_f64.pt" \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect float64=1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --out perf/of3t_ditmodel/MODEL_d56_retake.json \
  --sidecar-dir "$D/sidecar_d56_retake"
