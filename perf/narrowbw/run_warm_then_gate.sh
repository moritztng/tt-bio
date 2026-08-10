#!/bin/sh
# Waits for the running 298 aa sweep, then: warm double-visit cap sweeps (each cap arm appears
# twice; the SECOND visit is program-cache-warm and is the number of record, because the first
# visit pays a one-time program build inside the timed ttnn.linear call -- ~338 ms at the template
# site, measured as bw:8 362.32 vs the warm bw:16 24.41 at 512 aa), then the four parity gates.
set -x
cd /home/ttuser/.coworker/wt/protenix-trunk--z-narrowbw-512
WAITPID="$1"
while [ -n "$WAITPID" ] && kill -0 "$WAITPID" 2>/dev/null; do sleep 10; done
SP=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-narrowbw-512
export TT_MESH_GRAPH_DESC_PATH=$SP/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH=$PWD
PY=/home/ttuser/tt-bio-dev/env/bin/python3
WARM=on,bw:2,bw:2,on,bw:4,bw:4,on,bw:8,bw:8,on

$PY perf/narrowbw/nbw_arms.py --size 512 --arms $WARM \
    --out perf/narrowbw/nbw_512_warm_qb2c0.json
echo "WARM512_EXIT=$?"
$PY perf/narrowbw/nbw_arms.py --size 298 --arms $WARM \
    --out perf/narrowbw/nbw_298_warm_qb2c0.json
echo "WARM298_EXIT=$?"

for CAP in 1 8 4 2; do
  rm -f /tmp/nbw_mark_$CAP
  TTBIO_NARROW_PROJ_BW=$CAP TTBIO_NARROW_PROJ_BW_MARK=/tmp/nbw_mark_$CAP \
  $PY scripts/full_parity_gate.py --workers qb2:0 \
      --leg protenix-hsa-msa --leg protenix-prot-msa --leg protenix-ubq-msa \
      --workdir /tmp/nbw_gate_$CAP --out perf/narrowbw/gate_cap$CAP.json 2>&1 \
    | tee perf/narrowbw/gate_cap$CAP.log
  echo "GATE_CAP${CAP}_DONE"
  wc -l /tmp/nbw_mark_$CAP
done
echo "ALL_DONE"
