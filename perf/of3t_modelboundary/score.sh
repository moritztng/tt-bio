#!/usr/bin/env bash
# of3t-modelboundary: the model-scope assembly WITH the pairformer trunk in the union.
#
# `of3t-wholemodel/run_model.sh` carries the trunk as a --composed-term because its arms were
# driven by boundary_c64.pt. This row's arms are driven by boundary_n384.pt, the same capture
# uncropped, which IS the model's batch_step003 at crop 384 -- so the trunk goes in the union
# here, with --arm, and nothing is composed.
#
# CPU only. Every device arm it reads is already on disk.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-modelboundary
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
PIN=/home/ttuser/of3t_refprec/pinned_p175
O=/tmp/of3t/of3t-modelboundary
D=/home/ttuser/of3t_modelboundary
mkdir -p "$D"

OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py \
  --arm "shipped:diffusion=/home/ttuser/of3t_f64softmax/device_grads_043all_shipped.pt" \
  --arm "shipped:cond=/home/ttuser/of3t_direct/cond_grads_fp32.pt" \
  --arm "shipped:aux=/home/ttuser/of3t_auxgrad_logs/grads_M.pt" \
  --arm "shipped:msa=/home/ttuser/of3t_direct/msa_grads.pt" \
  --arm "shipped:pairformer_stack=$O/dev_CTRL_n384_nocaptures.pt" \
  --arm "renorm:diffusion=/home/ttuser/of3t_f64softmax/device_grads_043all_renorm.pt" \
  --arm "renorm:cond=/home/ttuser/of3t_wholemodel/cond_grads_renorm.pt" \
  --arm "renorm:aux=/home/ttuser/of3t_wholemodel/aux_grads_renorm.pt" \
  --arm "renorm:msa=/home/ttuser/of3t_wholemodel/msa_grads_renorm.pt" \
  --arm "renorm:pairformer_stack=$O/dev_RENORM_n384_nocaptures.pt" \
  --unwrap-key grads \
  --f64  /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
  --bf16 "$PIN/arm4_bf16_autocast/grads_f64.pt" \
  --f32  "$PIN/arm2_f32_upstream/grads_f64.pt" \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --expect float64=1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4 \
  --expect upstream_bf16=ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb \
  --expect upstream_f32=09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548 \
  --out perf/of3t_modelboundary/MODEL_withtrunk_n384.json \
  --sidecar-dir "$D/sidecar_modelboundary"
