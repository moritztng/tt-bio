#!/bin/bash
# Runs ON the box after box_bc_setup.sh. The BindCraft section of spd-kitcmp, on one A100:
#   stock BindCraft 2 (PacesaLab) on its shipped PD-L1 example, binder 80, campaign seed 101, 1 cold + 3 warm
#     trajectories, one at a time -- the same unit perf/spd/bench.py --model bindcraft2 reports on a TT chip;
#   the Anthropic kit's colabdesign kit (BindCraft 1 on ColabDesign 1.1.3, NOT like-for-like) in off, exact and fast,
#     on the same PD-L1 target (the two PDB files carry the identical 115 CA residues of chain A), binder 80, the
#     BindCraft 2 example's hotspots, one cold design per mode (seed 99, discarded) then 3 timed designs (seeds 0,1,2).
# results /root/kc/results/bindcraft/{stock,kit-<mode>}/...   markers ARMS-DONE
# GPU of this session (session.sh pushes /root/kc/gpu.env): the kit's --config and TORCH_CUDA_ARCH_LIST
KC_CFG=a100 KC_ARCH=8.0; [ -f /root/kc/gpu.env ] && . /root/kc/gpu.env
set -u
R=/root/kc/results/bindcraft; mkdir -p $R
export PYTHONHASHSEED=0 COLABDESIGN_PARAMS_DIR=/weights/af2 MAMBA_ROOT_PREFIX=/opt/mamba
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $R/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used --format=csv,noheader -l 1 > $R/smi.csv 2>&1 &
SMI=$!

# --- stock BindCraft 2 ----------------------------------------------------------------------------------------------
d=$R/stock; mkdir -p $d; date -u +%FT%T.%3NZ > $d/t_start
# bc2_gpu.py reads the 1 Hz clock log from its own --out; give it the one this script keeps
ln -sf $R/smi.csv $d/smi.csv
/opt/bc2-venv/bin/python /root/kc/bc2_gpu.py --bc2 /root/bc2 --out $d --seed ${BC2_SEED:-101} \
  --warm ${BC2_WARM:-3} --binder-len ${BLEN:-80} > $d/stdout.log 2> $d/stderr.log
echo $? > $d/rc; date -u +%FT%T.%3NZ > $d/t_end

# --- the kit's colabdesign (BindCraft 1) ------------------------------------------------------------------------------
K=/root/kit/colabdesign; E=/opt/mamba/envs/cdkit
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib
T=$K/stock/src/bindcraft/example/PDL1.pdb
A="--starting-pdb $T --chains A --binder-len ${BLEN:-80} --target-hotspot-residues ${HOTSPOTS:-54,56,66,115}"
[ -n "${BC_STOCK_ONLY:-}" ] && MODES=none
for m in ${MODES:-off exact fast}; do [ $m = none ] && break
  d=$R/kit-$m; mkdir -p $d; date -u +%FT%T.%3NZ > $d/t_start
  # seed 99 first: it pays the per-token-count XLA compile, and is discarded (run.sh warm takes no hotspots, so the
  # cold design goes through the same line as the timed ones)
  ( cd $K && bash run.sh design --config $KC_CFG --mode $m $A --seed 99 --out $d/cold ) > $d/cold.log 2>&1
  for s in ${SEEDS:-0 1 2}; do
    ( cd $K && bash run.sh design --config $KC_CFG --mode $m $A --seed $s --out $d/s$s ) > $d/design-s$s.log 2>&1
    echo "$s $?" >> $d/rc
  done
  date -u +%FT%T.%3NZ > $d/t_end
done
kill $SMI
echo ARMS-DONE > $R/ARMS-DONE
