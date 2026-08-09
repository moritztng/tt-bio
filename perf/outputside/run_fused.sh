#!/bin/bash
# Retry the fused A/B until a card is actually free. Other legs' jobs open chips they do
# not hold a lease for, so "my" card can be busy for a while.
cd /home/ttuser/.coworker/wt/perfwar-outputside-fusion || exit 1
MGD=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
for attempt in $(seq 1 40); do
  for card in 3 0 1 2; do
    echo "=== attempt $attempt card $card $(date -u +%H:%M:%S) ===" >> perf/outputside/fused_ab_runner.log
    TT_VISIBLE_DEVICES=$card TT_MESH_GRAPH_DESC_PATH=$MGD \
      TT_BIO_LEASE_HOLDER=worker:perfwar-outputside-fusion \
      PYTHONPATH=/home/ttuser/.coworker/wt/perfwar-outputside-fusion \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/outputside/fused_ab.py \
      --out perf/outputside/fused_ab_c64.json > perf/outputside/fused_ab_c64.log 2>&1
    if grep -q "bit-exact vs ttnn" perf/outputside/fused_ab_c64.log; then
      echo "DONE card $card" >> perf/outputside/fused_ab_runner.log
      exit 0
    fi
    if grep -q "Traceback" perf/outputside/fused_ab_c64.log && \
       ! grep -q "DeviceInUseError\|Custom fabric" perf/outputside/fused_ab_c64.log; then
      echo "REAL ERROR card $card" >> perf/outputside/fused_ab_runner.log
      exit 2
    fi
  done
  sleep 20
done
echo "GAVE UP" >> perf/outputside/fused_ab_runner.log
exit 3
