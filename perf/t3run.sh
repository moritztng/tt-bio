#!/bin/sh
# T3 (protenix-trunk--transition-norms) device run helper: qb2 card 1, single-chip pin.
WT=/home/ttuser/.coworker/wt/protenix-trunk--transition-norms
PY=/home/ttuser/tt-bio-dev/env/bin/python3
MGD=$($PY -c "import ttnn,os;print(os.path.dirname(ttnn.__file__))" 2>/dev/null | tail -1)/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
cd "$WT" || exit 1
exec env TT_VISIBLE_DEVICES=1 \
  TT_MESH_GRAPH_DESC_PATH="$MGD" \
  TT_BIO_LEASE_HOLDER=worker:protenix-trunk--transition-norms \
  PYTHONPATH="$WT" TT_METAL_LOGGER_LEVEL=FATAL \
  "$PY" "$@"
