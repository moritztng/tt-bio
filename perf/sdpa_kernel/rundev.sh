#!/bin/bash
# W9 device-run wrapper, qb2 card 1 (P300c needs the p150 mesh-graph descriptor for a single chip)
cd /home/ttuser/.coworker/wt/perfwar-sdpa-kernel
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:perfwar-sdpa-kernel
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH=/home/ttuser/.coworker/wt/perfwar-sdpa-kernel
exec /home/ttuser/tt-bio-dev/env/bin/python3 "$@"
