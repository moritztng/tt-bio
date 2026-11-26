#!/bin/bash
# The exact upstream command the R4T GPU training baseline times (state/r4t/rows/gpu-baseline.md, PROTOCOL).
#   r4t_train_cmd.sh <out_dir> <n_steps> [extra hydra overrides, e.g. an arm's mode knobs]
# Upstream's own `rfd4-train` and its canonical recipe (experiment=sota -> releases/flow/2026_10_19_hero_pretrain:
# crop 256 fixed padding, exploration_k 4, Poisson(3) recycles clamped [1,9] synced on global_step, bf16-mixed,
# torch.compile, cuEquivariance tri-mult, Muon + Adam, clip 1.0, EMA 0.999), initialised from the pinned release
# bundle. Only three things differ from a production launch, and none touches the step's work:
#   * the training data is upstream's own tiny test_dataset (PLINDER subset) instead of the 10-source mix, because
#     the mix is terabytes; the collate pads every batch to I=256 either way, so the network sees the same shapes;
#   * single GPU, no wandb, no checkpoint writing, no validation;
#   * StepTimer (r4t_step_timer.py) is added as a callback; it reads clocks and loss, nothing else.
# Env: DATA_PATH (holds test_dataset/), RFD4 (upstream checkout), PY (venv python), R4T_KC (dir with the timer).
set -eu
OUT=${1:?out dir} N=${2:?steps}; shift 2
RFD4=${RFD4:-/root/rfd4} PY=${PY:-/opt/rfd4-venv/bin/python} R4T_KC=${R4T_KC:-/root/kc}
mkdir -p "$OUT"
DROP=$(cd "$RFD4" && "$PY" - <<'EOF'
from hydra import compose, initialize_config_dir
import os
with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "configs"), version_base=None):
    cfg = compose(config_name="training_proteina", overrides=["experiment=sota"])
print(" ".join(f"~datasets.train.{k}" for k in cfg.datasets.train if k != "global_design_task_kwargs"))
EOF
)
cd "$RFD4"
# The dropped production sources interpolate this root; nothing reads it once they are gone.
export DATASET_STORE_ROOT=${DATASET_STORE_ROOT:-/nonexistent}
# shellcheck disable=SC2086
exec env PYTHONPATH="$R4T_KC${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/rfproteina/train.py \
  experiment=sota $DROP "~_hero_streaming" "+datasets/train@datasets.train.test_dataset=test_dataset" \
  hf_ckpt_path=nvidia/RFD4-Proteina hf_ckpt_revision=a06e2509c7033b010fc43614e9fd840303588502 pretrain_ckpt_path=null \
  +single=true +nolog=true log.log_wandb=false log.checkpoint=false \
  "opt.steps_per_epoch=$N" opt.max_epochs=1 ++opt.num_sanity_val_steps=0 ++opt.limit_val_batches=0 \
  "~metrics/refolding_oracles" \
  "+callbacks.r4t_timer._target_=r4t_step_timer.StepTimer" "+callbacks.r4t_timer.out=$OUT/steps.jsonl" \
  "++out_dir=$OUT/run" run_name_=r4t_gpu_baseline \
  "$@"
