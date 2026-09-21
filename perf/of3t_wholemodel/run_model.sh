#!/usr/bin/env bash
# The model-scope assembly. CPU only: every device arm it reads is already on disk.
#   run_model.sh <tag> <arm specs...>
#
# The pairformer trunk enters as a COMPOSED TERM, not as part of the union: its arms are driven
# by the captured 64-token boundary `boundary_c64.pt`, not by the model's own batch_step003 at
# crop 384, so concatenating its gradient with the rest would be concatenating gradients of two
# different inputs. Each term carries its OWN floor and its own r, because the per-section
# floors on this model span 7.7x and a borrowed floor is wrong on most of it (A28/A27).
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
PIN=/home/ttuser/of3t_refprec/pinned_p175
TAG=${1:?tag}; shift
ARGS=()
for a in "$@"; do ARGS+=(--arm "$a"); done
B="captured 64-token boundary boundary_c64.pt (56 real tokens of 64), not batch_step003"
OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py "${ARGS[@]}" \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 "$PIN/arm4_bf16_autocast/grads_f64.pt" \
  --f32  "$PIN/arm2_f32_upstream/grads_f64.pt" \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --composed-term "shipped:pairformer_stack=5.8282,8.713525811931007,0.37393553950708647,1.029909337085664,$B" \
  --composed-term "renorm:pairformer_stack=5.8282,0.47560108602437734,0.37393553950708647,1.029909337085664,$B" \
  --composed-term "renormf64:pairformer_stack=5.8282,0.47560108602437734,0.37393553950708647,1.029909337085664,$B (host f64 softmax bit-identical to renorm here, 0 calls served)" \
  --composed-term "break:pairformer_stack=5.8282,8.080414910687447,0.37393553950708647,1.029909337085664,$B" \
  --out "perf/of3t_wholemodel/MODEL_${TAG}.json" \
  --sidecar-dir "/home/ttuser/of3t_wholemodel/sidecar_${TAG}"
