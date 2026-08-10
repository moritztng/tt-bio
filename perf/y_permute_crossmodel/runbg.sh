#!/bin/bash
# Launch the boltzgen census/A-B leg on qb2 card 2.
WT=/home/ttuser/.coworker/wt/protenix-trunk--y-permute-crossmodel
MESH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_MESH_GRAPH_DESC_PATH="$MESH"
export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--y-permute-crossmodel
export PYTHONPATH="$WT"
exec "$PY" perf/y_permute_crossmodel/boltzgen_ab.py "$@"
