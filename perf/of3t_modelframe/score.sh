#!/usr/bin/env bash
# of3t-modelframe step 4: the graded artifact, rescored with pairformer_stack taken from the arm
# on the model's own boundary.
#
# This is `perf/of3t_covadopt/score.sh` -- the producer of
# MODEL_withtrunk_composed3660_n384.json, the artifact the charter gate reads -- with one --arm
# path changed and a different --out. The five other arms, the three pinned reference digests,
# the section table and the unwrap key are its own. CPU only, no card is opened.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-modelframe
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=/tmp/of3t/of3t-modelframe
COV=/tmp/of3t/of3t-covadopt
TRUNK=${1:-$O/dev_RENORM_model_n384_nocaptures.pt}
[ -s "$TRUNK" ] || { echo "missing trunk arm $TRUNK -- run runarm.sh first"; exit 2; }
mkdir -p "$O"

OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py \
  --arm "renorm:diffusion=$COV/device_grads_rc_refatom_on.pt" \
  --arm "renorm:input_embedder=$COV/ie_grads_f64_renorm_on.pt" \
  --arm "renorm:cond=/home/ttuser/of3t_wholemodel/cond_grads_renorm.pt" \
  --arm "renorm:aux=/home/ttuser/of3t_wholemodel/aux_grads_renorm.pt" \
  --arm "renorm:msa=/home/ttuser/of3t_wholemodel/msa_grads_renorm.pt" \
  --arm "renorm:pairformer_stack=$TRUNK" \
  --unwrap-key grads \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 /home/ttuser/of3t_refprec/pinned_p175/arm4_bf16_autocast/grads_f64.pt \
  --f32  /home/ttuser/of3t_refprec/pinned_p175/arm2_f32_upstream/grads_f64.pt \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect float64=1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --out perf/of3t_modelframe/MODEL_FRAMEMATCHED_composed3660_n384.json \
  --sidecar-dir "$O/sidecar_framematched" 2>&1 | tail -30
echo "=== model_scope exit ${PIPESTATUS[0]} ==="

"$PY" perf/of3t_modelframe/clause.py \
  --published perf/of3t_modelboundary/MODEL_withtrunk_composed3660_n384.json \
  --rescored  perf/of3t_modelframe/MODEL_FRAMEMATCHED_composed3660_n384.json \
  --out       perf/of3t_modelframe/CLAUSE.json
