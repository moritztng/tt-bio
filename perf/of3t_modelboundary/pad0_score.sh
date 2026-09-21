#!/usr/bin/env bash
# of3t-modelboundary: is the trunk's model-scope excess the PAD region?
#
# The n384 boundary is 56 real tokens of 384. Measured on the boundary itself: s_in's pad part
# carries norm 1.0325e4 against the real part's 5.6235e3, and z_in's real-by-real block is
# 6.4264e4 of a 1.6158e7 total -- the pair track is 99.6 % pad by norm. A weight gradient sums
# over every row, pad included, so unlike a forward it cannot be masked afterwards.
#
# So: the same RENORM arm with the pad rows and columns zeroed on the way in, scored through the
# same scorer, same float64 reference, same denominator. renorm reads 2.159527 on the 2,736. If
# pad0 collapses toward upstream's own bf16 floor of 0.3147698, the pad is the mechanism. If it
# does not, the disagreement is on the 56 real tokens.
#
# CPU only, and it writes its OWN artifact -- MODEL_withtrunk_n384.json is committed and A33
# keeps this row off files another row's charter reads.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-modelboundary
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
PIN=/home/ttuser/of3t_refprec/pinned_p175
O=/tmp/of3t/of3t-modelboundary
D=/home/ttuser/of3t_modelboundary
mkdir -p "$D"

OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py \
  --arm "renorm:diffusion=/home/ttuser/of3t_f64softmax/device_grads_043all_renorm.pt" \
  --arm "renorm:cond=/home/ttuser/of3t_wholemodel/cond_grads_renorm.pt" \
  --arm "renorm:aux=/home/ttuser/of3t_wholemodel/aux_grads_renorm.pt" \
  --arm "renorm:msa=/home/ttuser/of3t_wholemodel/msa_grads_renorm.pt" \
  --arm "renorm:pairformer_stack=$O/dev_RENORM_n384_nocaptures.pt" \
  --arm "pad0:diffusion=/home/ttuser/of3t_f64softmax/device_grads_043all_renorm.pt" \
  --arm "pad0:cond=/home/ttuser/of3t_wholemodel/cond_grads_renorm.pt" \
  --arm "pad0:aux=/home/ttuser/of3t_wholemodel/aux_grads_renorm.pt" \
  --arm "pad0:msa=/home/ttuser/of3t_wholemodel/msa_grads_renorm.pt" \
  --arm "pad0:pairformer_stack=$O/dev_RENORM_n384_pad0.pt" \
  --unwrap-key grads \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 "$PIN/arm4_bf16_autocast/grads_f64.pt" \
  --f32  "$PIN/arm2_f32_upstream/grads_f64.pt" \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect float64=1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --out perf/of3t_modelboundary/MODEL_pad0_n384.json \
  --sidecar-dir "$D/sidecar_pad0"
echo "SCORE_EXIT=$?"
