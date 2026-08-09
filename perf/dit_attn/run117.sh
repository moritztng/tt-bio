#!/bin/bash
# 117 aa non-regression: core_grid on a 2-tile-wide output can invert at small N.
set -u
WT=/home/ttuser/.coworker/wt/perfwar-dit-attention-fusion
BASE="$WT/.base"
BASE_REF=${BASE_REF:-origin/main}   # the baseline arm folds whatever this points at
P=/home/ttuser/tt-bio-dev/env/bin/python3

if [ ! -d "$BASE/tt_bio" ]; then
  git -C "$WT" worktree add -q --detach "$BASE" "$BASE_REF" || exit 1
  mkdir -p "$BASE/perf/dit_attn"   # this dir only exists on the working branch
  cp "$WT/perf/dit_attn/ab_fold.py" "$BASE/perf/dit_attn/ab_fold.py"
fi

export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:perfwar-dit-attention-fusion

for arm in before after; do
  if [ "$arm" = before ]; then D="$BASE"; else D="$WT"; fi
  echo "########## 117aa / $arm ##########"
  cd "$D" || exit 1
  PYTHONPATH="$D" stdbuf -oL -eL "$P" -u perf/dit_attn/ab_fold.py --model protenix-v2 \
      --size 117 --repeat 3 --tag "117_$arm" \
      --out "$WT/perf/dit_attn/fold117_protenix-v2_$arm.json" 2>&1 \
    | stdbuf -oL grep -avE "info |Fabric|topology|Degree|Config\{|DEBUG|loguru|warning|Always|SiliconDriver|Adjacency|Total nodes|^ ---|backtrace"
done
echo RUN117_DONE
