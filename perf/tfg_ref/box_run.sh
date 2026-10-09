#!/usr/bin/env bash
# Run upstream OpenDDE 1.2.0 on panel targets: one invocation per (condition, seed), all targets in one input list.
# usage: box_run.sh PANEL_DIR RUN_DIR TARGETS_FILE "SEEDS" "CONDS"   e.g. "101 102" "unconstrained contact pocket"
# Unconstrained runs the standard sampler (--use_tfg_guidance false); contact/pocket run TFG with upstream's defaults.
set -euo pipefail
PANEL=$(realpath "$1"); RUN=$(realpath -m "$2"); TARGETS=$(realpath "$3"); SEEDS=$4; CONDS=$5
CKPT=${OPENDDE_CKPT:-/root/opendde_data/checkpoint/opendde_abag.pt}
mkdir -p "$RUN"
cd "$PANEL"
for seed in $SEEDS; do
  for cond in $CONDS; do
    out=$RUN/$cond/seed_$seed
    [ -f "$out/.done" ] && continue
    mkdir -p "$out"
    python3 - "$TARGETS" "$cond" "$out/input.json" <<'PY'
import json, sys
tids = [t for t in open(sys.argv[1]).read().split() if t]
jobs = [json.load(open(f"{t}/{t}_{sys.argv[2]}.json"))[0] for t in tids]
json.dump(jobs, open(sys.argv[3], "w"))
PY
    tfg=false; [ "$cond" != unconstrained ] && tfg=true
    t0=$(date +%s)
    opendde pred -i "$out/input.json" -o "$out" --load_checkpoint_path "$CKPT" \
      --seeds "$seed" --sample 5 --cycle 10 --step 200 --dtype bf16 --enable_tf32 false \
      --use_msa true --use_template false --use_tfg_guidance $tfg > "$out/log.txt" 2>&1
    echo "$cond seed=$seed targets=$(wc -w < "$TARGETS") wall_s=$(( $(date +%s) - t0 )) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "$RUN/timing.log"
    touch "$out/.done"
  done
done
