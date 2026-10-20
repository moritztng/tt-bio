#!/bin/bash
# bc4-bet-triatt chain 1 on the first free qb2 chip: per-call A/B of the triangle-attention backward
# (graded vs float64), its per-phase zones at 224 under the Tracy build, then a census of this tree's
# BindCraft 2 iteration with all 48 Evoformer tapes resident (main's K=48, which auto only picks on a
# p150a). AICLK + load at 1 Hz in aiclk.tsv.
#   setsid nohup bash perf/bc4_triatt/chain1.sh > /dev/null 2>&1 &
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=${OUT:-$WT/_bc4out/chain1_$(git rev-parse --short HEAD)}; mkdir -p "$D"; echo $$ > "$D/pid"
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"; hostname > "$D/host"
PY=$HOME/bcx_e2e_venv/bin/python
PARAMS=$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz
CARD=
while [ -z "$CARD" ]; do
  for c in ${CHIPS:-3 2 1 0}; do
    exec 9>"$HOME/spd_qb2_chip$c.lock"
    if flock -n 9; then CARD=$c; break; fi
    exec 9>&-
  done
  [ -z "$CARD" ] && sleep 30
done
echo "card $CARD start $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> "$D/card"
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-triatt
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk 2>/dev/null)" \
    "$(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
step() { local n=$1; shift; echo "$n start $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> "$D/card"
  "$@" > "$D/$n.log" 2>&1; echo "$n rc=$? end $(date +%s)" >> "$D/card"; }
ARMS="base:EXP_21F=1;exact:EXP_21F=0;approx:EXP_APPROX=1;hifi2:EXP_21F=1,FIDELITY=HiFi2;lofi:EXP_21F=1,FIDELITY=LoFi;approx_hifi2:EXP_APPROX=1,FIDELITY=HiFi2"
V2ARMS="base:EXP_21F=1;v2:EXP_21F=1,V2=1;v2_approx:EXP_APPROX=1,V2=1;v2_approx_hifi2:EXP_APPROX=1,V2=1,FIDELITY=HiFi2"
step bench224 timeout -s INT -k 60 900 $PY perf/bc4_triatt/bw_bench.py --n 224 --arms "$ARMS" --reps 20 --out "$D/bench224.json"
step bench224v2 timeout -s INT -k 60 900 $PY perf/bc4_triatt/bw_bench.py --n 224 --arms "$V2ARMS" --reps 20 --out "$D/bench224v2.json"
step bench224b timeout -s INT -k 60 900 $PY perf/bc4_triatt/bw_bench.py --n 224 --bias-std 2 --arms "$ARMS;${V2ARMS#base:EXP_21F=1;}" --reps 4 --out "$D/bench224b.json"
# Zones: the Tracy build of the same ttnn (tt-metal-b2z v0.68.0), one profiled call per arm.
M=/home/ttuser/tt-metal-b2z
step zones224 env TT_METAL_HOME=$M PYTHONPATH=$WT:$M/ttnn:$M/tools:$M TT_METAL_DEVICE_PROFILER=1 \
  TT_METAL_PROFILER_DIR=$D/prof TT_BIO_TRIATT_BW_ZONES=1 PATH=$HOME/bcx_e2e_venv/bin:$PATH \
  timeout -s INT -k 60 900 $PY perf/bc4_triatt/bw_bench.py --n 224 --lead 27 --zones \
  --arms "base:EXP_21F=1;approx:EXP_APPROX=1;v2:EXP_21F=1,V2=1;v2_approx:EXP_APPROX=1,V2=1" --out "$D/zones224.json"
step bench288 timeout -s INT -k 60 900 $PY perf/bc4_triatt/bw_bench.py --n 288 --arms "base:EXP_21F=1;v2:EXP_21F=1,V2=1;v2_approx:EXP_APPROX=1,V2=1" --reps 10 --out "$D/bench288.json"
step census env TT_BIO_BC2_RESIDENT_BLOCKS=48 timeout -s INT -k 120 3000 $PY perf/spd/bc2_evo_devprof.py \
  --params "$PARAMS" --out "$D/census" --warm 2 --reps 3 --census
kill $P; touch "$D/done"
