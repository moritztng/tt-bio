#!/usr/bin/env bash
# of3t-recutfin job 1: re-emit the composed artifact WITH the per-scope injection stamp.
#
# Same instrument, same six arms, same three pinned references, same section table as the run
# of3t-recut published. The only difference is that the composer now records which functional
# produced each scope. So every number the artifact already carried must come back bit-identical:
# a difference in any of them is a finding, not a fix, and restamp_control.py is what says so
# before the file is moved into place.
#
# The new file is written OUTSIDE the tree and only installed once that control passes.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-recutfin
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=/home/ttuser/of3t_recutfin
COV=/home/ttuser/of3t_covadopt
TRUNK=/home/ttuser/of3t_recut/dev_RENORM_model_n384_external.pt
COT=/home/ttuser/of3t_recut/cot_external.pt
NEW=$O/MODEL_RECUT_composed3660_n384.json
mkdir -p "$O"

OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py \
  --arm "renorm:diffusion=$COV/device_grads_rc_refatom_on.pt" \
  --arm "renorm:input_embedder=$COV/ie_grads_f64_renorm_on.pt" \
  --arm "renorm:cond=/home/ttuser/of3t_wholemodel/cond_grads_renorm.pt" \
  --arm "renorm:aux=/home/ttuser/of3t_wholemodel/aux_grads_renorm.pt" \
  --arm "renorm:msa=/home/ttuser/of3t_wholemodel/msa_grads_renorm.pt" \
  --arm "renorm:pairformer_stack=$TRUNK" \
  --injection "renorm:diffusion=not_injected" \
  --injection "renorm:input_embedder=not_injected" \
  --injection "renorm:cond=not_injected" \
  --injection "renorm:aux=not_injected" \
  --injection "renorm:msa=not_injected" \
  --injection "renorm:pairformer_stack=graph-cut-external" \
  --injection-correction "renorm:pairformer_stack=$COT" \
  --unwrap-key grads \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 /home/ttuser/of3t_refprec/pinned_p175/arm4_bf16_autocast/grads_f64.pt \
  --f32  /home/ttuser/of3t_refprec/pinned_p175/arm2_f32_upstream/grads_f64.pt \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect float64=1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --out "$NEW" --sidecar-dir "$O/sidecar_restamp" 2>&1 | tail -25
RC=${PIPESTATUS[0]}
echo "=== model_scope exit $RC out $NEW ==="
[ "$RC" -eq 0 ] || exit "$RC"

"$PY" perf/of3t_recutfin/restamp_control.py --new "$NEW" \
  --banked perf/of3t_recut/MODEL_RECUT_composed3660_n384.json --install
echo "=== restamp_control exit $? ==="
