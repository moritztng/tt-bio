#!/bin/bash
# Card 1 on qb2 is a LONE p300 chip: a CUSTOM topology to tt-metal that will not open without a
# 1x1 graph. tt-bio sets this per worker itself; a bare probe has to do it by hand
# (memory qb2-p300c-dispatch-granularity-is-a-board-pair).
cd /home/ttuser/.coworker/wt/bcx-bfp8
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-bfp8
export PYTHONPATH=/home/ttuser/.coworker/wt/bcx-bfp8
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/bcx_bfp8/roofs.py "$@"
