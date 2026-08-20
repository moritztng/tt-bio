#!/bin/sh
cd /home/ttuser/.coworker/wt/opendde-size-generality-l1-work-split || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:opendde-size-generality-l1-work-split
export PYTHONPATH=/home/ttuser/.coworker/wt/opendde-size-generality-l1-work-split
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
exec /home/ttuser/tt-bio-dev/env/bin/python -u perf/other512/fold_ab_multi.py   --model opendde --sizes 1024 --arms on,qpercore,on   --out perf/oddel1/fold_ab_1024_qb2c0.json
