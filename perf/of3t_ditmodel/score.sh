#!/usr/bin/env bash
# of3t-ditmodel: model scope WITH the pairformer trunk in the union, per D174 arm.
#
# of3t-modelboundary's score.sh with one difference: every non-trunk scope is this row's own
# re-take on today's tree (the d56on arms from the first half of this row), so the union is one
# tree and the only variable between the arms below is TT_BIO_MASK_TRANS. CPU only.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-ditmodel
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
PIN=/home/ttuser/of3t_refprec/pinned_p175
D=/tmp/of3t/of3t-ditmodel

ARGS=()
for a in maskoff maskon maskones; do
  T=$(echo "$a" | tr 'a-z' 'A-Z')
  [ -f "$D/dev_${T}_n384.pt" ] || { echo "missing $D/dev_${T}_n384.pt"; continue; }
  ARGS+=(--arm "$a:diffusion=$D/device_grads_043all_d56on.pt")
  ARGS+=(--arm "$a:cond=$D/cond_grads_d56on.pt")
  ARGS+=(--arm "$a:aux=$D/aux_grads_d56on.pt")
  ARGS+=(--arm "$a:msa=$D/msa_grads_d56on.pt")
  ARGS+=(--arm "$a:pairformer_stack=$D/dev_${T}_n384.pt")
done

OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py "${ARGS[@]}" \
  --unwrap-key grads \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 "$PIN/arm4_bf16_autocast/grads_f64.pt" \
  --f32  "$PIN/arm2_f32_upstream/grads_f64.pt" \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect float64=1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --out perf/of3t_ditmodel/MODEL_D174_n384.json \
  --sidecar-dir "$D/sidecar_d174"
