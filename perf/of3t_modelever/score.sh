#!/usr/bin/env bash
# of3t-modelever: compose a trunk arm into the graded 3,660-tensor model artifact and read the
# clause off it. perf/of3t_recut/score.sh with the trunk path and outputs changed and nothing
# else: the five other arms, the three pinned reference digests, the section table and the
# unwrap key are its own. CPU only, no card is opened.
#
#   score.sh <TAG> <trunk.pt>     -> perf/of3t_modelever/MODEL_<TAG>_composed3660_n384.json
#                                    perf/of3t_modelever/CLAUSE_<TAG>.json
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-modelever
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=/home/ttuser/of3t_modelever
COV=/home/ttuser/of3t_covadopt
TAG=${1:?usage: score.sh TAG trunk.pt}; TRUNK=${2:?usage: score.sh TAG trunk.pt}
[ -s "$TRUNK" ] || { echo "missing trunk arm $TRUNK"; exit 2; }
OUT=perf/of3t_modelever/MODEL_${TAG}_composed3660_n384.json

OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py \
  --arm "renorm:diffusion=$COV/device_grads_rc_refatom_on.pt" \
  --arm "renorm:input_embedder=$COV/ie_grads_f64_renorm_on.pt" \
  --arm "renorm:cond=/home/ttuser/of3t_wholemodel/cond_grads_renorm.pt" \
  --arm "renorm:aux=/home/ttuser/of3t_wholemodel/aux_grads_renorm.pt" \
  --arm "renorm:msa=/home/ttuser/of3t_wholemodel/msa_grads_renorm.pt" \
  --arm "renorm:pairformer_stack=$TRUNK" \
  --injection renorm:diffusion=not_injected --injection renorm:input_embedder=not_injected \
  --injection renorm:cond=not_injected --injection renorm:aux=not_injected \
  --injection renorm:msa=not_injected --injection renorm:pairformer_stack=graph-cut-external \
  --injection-correction renorm:pairformer_stack=/home/ttuser/of3t_recut/cot_external.pt \
  --unwrap-key grads \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 /home/ttuser/of3t_refprec/pinned_p175/arm4_bf16_autocast/grads_f64.pt \
  --f32  /home/ttuser/of3t_refprec/pinned_p175/arm2_f32_upstream/grads_f64.pt \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect float64=1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --out "$OUT" --sidecar-dir "$O/sidecar_$TAG" 2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "=== model_scope exit $rc tag $TAG out $OUT ==="
[ "$rc" = 0 ] || exit "$rc"
"$PY" perf/of3t_modelframe/clause.py \
  --published perf/of3t_modelboundary/MODEL_withtrunk_composed3660_n384.json \
  --rescored "$OUT" --out "perf/of3t_modelever/CLAUSE_${TAG}.json"
