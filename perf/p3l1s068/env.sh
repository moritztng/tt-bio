# one place for the card, the wheel and the worktree, so no run can silently take the wrong one
WT=/home/ttuser/.coworker/wt/protenix-trunk--p3-l1-source-068
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-l1-source-068
export PYTHONPATH=$WT
# a single P300 chip cannot be opened without a 1x1 mesh graph descriptor (charter 4.8)
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
QUIET="info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning"
