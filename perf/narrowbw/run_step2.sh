#!/bin/sh
set -x
cd /home/ttuser/.coworker/wt/protenix-trunk--z-narrowbw-512
SP=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-narrowbw-512
export TT_MESH_GRAPH_DESC_PATH=$SP/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH=$PWD
PY=/home/ttuser/tt-bio-dev/env/bin/python3
$PY perf/narrowbw/nbw_arms.py --size 512 \
    --arms on,off:narrowbw,on,bw:8,on,bw:4,on,bw:2,on,bw:16,on,bw:8+pairbw1,on \
    --out perf/narrowbw/nbw_512_qb2c0.json
echo "STEP2_EXIT=$?"
