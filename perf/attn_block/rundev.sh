#!/bin/bash
# W6 device-run wrapper on qb2 card 1 (P300c: needs the p150 mesh-graph descriptor for a single chip)
cd /home/ttuser/.coworker/wt/perfwar-attention-block-fusion
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:perfwar-attention-block-fusion
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH=/home/ttuser/.coworker/wt/perfwar-attention-block-fusion
exec /home/ttuser/tt-bio-dev/env/bin/python3 "$@"
