#!/bin/bash
cd /home/ttuser/.coworker/wt/perfwar-outputside-fusion || exit 1
MGD=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
run() {
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    for card in 3 1 2 0; do
      TT_VISIBLE_DEVICES=$card TT_MESH_GRAPH_DESC_PATH=$MGD \
        TT_BIO_LEASE_HOLDER=worker:perfwar-outputside-fusion \
        PYTHONPATH=/home/ttuser/.coworker/wt/perfwar-outputside-fusion \
        /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/outputside/fused_ab.py \
        --hidden "$1" --out "perf/outputside/stride_$2.json" \
        > "perf/outputside/stride_$2.log" 2>&1
      if grep -q "^best rbatch" "perf/outputside/stride_$2.log"; then
        echo "DONE $2 card $card" >> perf/outputside/run6.log; return 0
      fi
      if grep -q "Traceback" "perf/outputside/stride_$2.log" && \
         ! grep -q "DeviceInUseError\|Custom fabric" "perf/outputside/stride_$2.log"; then
        echo "REAL ERROR $2 card $card" >> perf/outputside/run6.log; return 2
      fi
    done
    sleep 15
  done
  echo "GAVE UP $2" >> perf/outputside/run6.log; return 3
}
run 256 pv2 || exit $?
echo "ALL DONE" >> perf/outputside/run6.log
