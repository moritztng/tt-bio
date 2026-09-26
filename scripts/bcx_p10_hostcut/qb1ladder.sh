#!/bin/bash
cd ~/bcx_hostcut
PY=/home/ttuser/bcx_e2e_venv/bin/python3
mkdir -p out
for c in 2 4 8 16 32; do
  taskset -c 0-$((c-1)) $PY -u scripts/bcx_p10_hostcut/hostcores.py \
    --ref /home/ttuser/bcx_structmod_art/ref_n288.npz --bc2 /home/ttuser/bcx_e2e/bc2 \
    --params /home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz \
    --tag "qb1-cores$c" --reps 6 --out out/qb1-cores$c.json >/dev/null 2>&1
  $PY -c "
import json
d=json.load(open('out/qb1-cores$c.json')); f,b=d['forward'],d['forward_and_backward']
print('aff %2d  fwd %6.3f s %4.2f cores | fwd+bwd %6.3f s %4.2f cores | load1 %5.2f | %s MHz'
      % (d['affinity_cores'], f['median_wall_s'], f['median_cores'],
         b['median_wall_s'], b['median_cores'], b['median_load1'], b.get('median_mhz')))
"
done
