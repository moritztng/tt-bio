#!/bin/bash
# One model, both arms, same card, back to back. The baseline arm runs out of a
# detached worktree at the pre-edit commit (tt_bio/ there is identical to origin/main),
# so nothing is ever swapped inside the working branch.
set -u
MODEL="${1:?model}"
REP="${2:-3}"
WT=/home/ttuser/.coworker/wt/perfwar-dit-attention-fusion
BASE="$WT/.base"
P=/home/ttuser/tt-bio-dev/env/bin/python3

export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:perfwar-dit-attention-fusion

run() {  # dir tag
  echo "########## $MODEL / $2 ##########"
  cd "$1" || return 1
  PYTHONPATH="$1" stdbuf -oL -eL "$P" -u perf/dit_attn/ab_fold.py --model "$MODEL" \
      --repeat "$REP" --tag "$2" --out "$WT/perf/dit_attn/fold_${MODEL}_$2.json" 2>&1 \
    | stdbuf -oL grep -avE "info |Fabric|topology|Degree|Config\{|DEBUG|loguru|warning|Always|SiliconDriver|Adjacency|Total nodes|^ ---|backtrace"
  return "${PIPESTATUS[0]}"
}

run "$BASE" before || echo "BEFORE_FAILED"
run "$WT" after || echo "AFTER_FAILED"
cd "$WT" && git status --short -- tt_bio/
echo "RUN_AB_DONE"
