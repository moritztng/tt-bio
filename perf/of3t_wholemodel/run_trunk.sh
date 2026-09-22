#!/usr/bin/env bash
# The pairformer trunk, scored on ITS OWN captured boundary. This scope cannot enter the model
# union -- its arms are driven by `boundary_c64.pt`, 56 real tokens of 64, not by the model's
# batch_step003 at crop 384 -- so it is scored here and carried into the model statement as a
# composed term with its boundary named.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
A=/home/ttuser/of3t_apbgrad
G=/home/ttuser/of3t_trunkg043
OMP_NUM_THREADS=8 nice -n 10 "$PY" perf/of3t_wholemodel/model_scope.py \
  --arm "shipped_softmax_bw:pairformer_stack=$A/dev_scope_CTRL_c64.pt" \
  --arm "renorm:pairformer_stack=$A/dev_scope_RENORM_c64.pt" \
  --arm "break:pairformer_stack=$A/dev_scope_BREAK_c64.pt" \
  --unwrap-key grads --scope-only \
  --f64  "$G/ref_f64_c64.pt" \
  --bf16 "$A/ref_bf16auto_c64.pt" \
  --f32  "$G/ref_f32_c64.pt" \
  --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
  --out perf/of3t_wholemodel/TRUNK_c64.json \
  --sidecar-dir /home/ttuser/of3t_wholemodel/sidecar_trunk
