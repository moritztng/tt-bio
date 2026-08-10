#!/bin/sh
# p2-alignment run helper: qb2 card 0 (board 007 chip 0), single-chip mesh override.
WT=/home/ttuser/.coworker/wt/protenix-trunk--p2-alignment
export TT_VISIBLE_DEVICES=0
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p2-alignment
export PYTHONPATH=$WT
cd $WT
exec /home/ttuser/tt-bio-dev/env/bin/python3 "$@"
