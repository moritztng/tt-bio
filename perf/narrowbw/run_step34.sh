#!/bin/sh
# Waits for the step-2 sweep to exit, then runs step 3 (298 aa arms) and step 4 (parity gates).
# Rooted in this worktree on purpose (fleet hygiene defers removal while a live process is here).
set -x
cd /home/ttuser/.coworker/wt/protenix-trunk--z-narrowbw-512
WAITPID="$1"
while [ -n "$WAITPID" ] && kill -0 "$WAITPID" 2>/dev/null; do sleep 10; done
SP=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-narrowbw-512
export TT_MESH_GRAPH_DESC_PATH=$SP/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH=$PWD
PY=/home/ttuser/tt-bio-dev/env/bin/python3

$PY perf/narrowbw/nbw_arms.py --size 298 \
    --arms on,off:narrowbw,on,bw:8,on,bw:4,on,bw:2,on,bw:16,on \
    --out perf/narrowbw/nbw_298_qb2c0.json
echo "STEP3_EXIT=$?"

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
echo "STEP4_ALL_DONE"
