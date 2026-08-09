#!/bin/bash
# One model, both arms, same card, same session. Restores the worktree on any exit.
set -u
MODEL="${1:?model}"
REP="${2:-3}"
WT=/home/ttuser/.coworker/wt/perfwar-dit-attention-fusion
BASE=638fd2f9          # tt_bio/ identical to origin/main (verified with git diff)
P=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1

restore() { cd "$WT" && git checkout -q HEAD -- tt_bio/tenstorrent.py tt_bio/protenix.py; }
trap restore EXIT

# wait for card 0 (the card the roofs in the state doc were measured on)
for i in $(seq 1 120); do
  PID=$(python3 -c "import json;d=json.load(open('/home/ttuser/.coworker/state/leases/tt-quietbox2-card0.json'));print('' if d.get('released') else d.get('pid',''))" 2>/dev/null)
  if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then echo "card 0 free after ${i}0s"; break; fi
  sleep 10
done

export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:perfwar-dit-attention-fusion
export PYTHONPATH="$WT"

run() {  # tag
  echo "########## $MODEL / $1 ##########"
  stdbuf -oL -eL "$P" -u perf/dit_attn/ab_fold.py --model "$MODEL" --repeat "$REP" \
      --tag "$1" --out "perf/dit_attn/fold_${MODEL}_$1.json" 2>&1 \
    | stdbuf -oL grep -avE "info |Fabric|topology|Degree|Config\{|DEBUG|loguru|warning|Always|SiliconDriver|Adjacency|Total nodes|^ ---|backtrace"
  return "${PIPESTATUS[0]}"
}

restore
run after || echo "AFTER_FAILED"
git checkout -q "$BASE" -- tt_bio/tenstorrent.py tt_bio/protenix.py
run before || echo "BEFORE_FAILED"
restore
git status --short
