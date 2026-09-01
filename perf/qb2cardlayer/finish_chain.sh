#!/usr/bin/env bash
# finish_chain.sh — the tail of the card-layer reseed: boltzgen (the one leg the first pass did
# not reach), then one clean gate run over all thirteen cells this task and its parent touched.
# Runs detached with its cwd inside this worktree, per worker.sh step 5.
set -u
WT=/home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed
O="$WT/perf/qb2cardlayer"
cd "$WT"
exec >>"$O/finish_chain.out" 2>&1
echo "=== finish chain start $(date -u +%FT%TZ) ==="

# 1. wait out the boltzgen draw campaign the previous pass left running
while kill -0 "${CAMPAIGN_PID:-0}" 2>/dev/null; do sleep 20; done
echo "=== campaign pid ${CAMPAIGN_PID:-none} exited $(date -u +%FT%TZ) ==="
bash "$O/index.sh" >/dev/null
bash "$O/tally.sh"

# 2. boltzgen note, composed from the draws rather than typed. The retired-box control is the
#    v0.7.2 gate of 2026-08-28, 0.021410 designs/s (state/tt-bio-release-v0-7-2.md).
NOTE=$(python3 "$O/boltzgen_note.py")
echo "--- note ---"; echo "$NOTE"; echo "------------"

echo "=== reseed boltzgen $(date -u +%FT%TZ) ==="
bash "$O/reseed.sh" boltzgen "$NOTE" 0

# 3. one clean gate pass over every cell this task (8, card layer) and its parent (5, machine
#    layer) reseeded, in a single process so each model is measured against the file as committed.
echo "=== verify 13 $(date -u +%FT%TZ) ==="
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH="$WT" TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:qb2-card-layer-baseline-reseed
bash /home/ttuser/.coworker/scripts/benchlock.sh cardlayer-verify13 -- \
  /home/ttuser/tt-bio-dev/env/bin/python scripts/perf_regression.py \
    --model boltz2 --model esmfold2 --model esmfold2-fast --model esmc-300m \
    --model esmc-600m --model esmc-6b --model boltzgen --model boltz2-affinity \
    --model saprot-650m --model opendde-abag --model rfd3 --model openfold3 --model pxdesign
echo "verify13 rc=$?"
echo "=== finish chain done $(date -u +%FT%TZ) ==="
