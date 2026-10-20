#!/bin/bash
# bc4-bet-elem chain 1: the elementwise floor of the BindCraft 2 Evoformer, measured in situ by ablation.
# Arms are separate bc2_evo_devprof processes (the ablations are compile-time defines), alternated A/X/A so
# drift falls on both sides. Each ablation removes ONE stage and computes wrong values on purpose; only its
# seam time means anything. Then prims.py (L1-resident primitive rates, serial check). AICLK + load at 1 Hz.
#   CARD=<n> bash perf/bc4_elem/chain1.sh     (detached; read with perf/bc4_elem/read1.py <dir>)
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=${OUT:-$WT/_bc4out/elem1_$(git rev-parse --short HEAD)}; mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
PARAMS=${PARAMS:-$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz}
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-$CARD} TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bc4-bet-elem}
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"; hostname > "$D/host"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk 2>/dev/null)" \
    "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" & P=$!
arm() {  # name, env assignments...
  local n=$1; shift
  echo "$n start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/arms"
  env "$@" timeout -s INT -k 60 1500 "$PY" perf/spd/bc2_evo_devprof.py --params "$PARAMS" --out "$D/$n" \
      --warm 2 --reps ${REPS:-6} > "$D/$n.log" 2>&1
  echo "$n end $(date +%s) rc=$?" >> "$D/arms"
}
for r in 1 2; do
  arm base$r     X=0
  arm fexp0_$r   TT_BIO_TRIATT_ABLATE=EXP
  arm bexp0_$r   TT_BIO_TRIATT_BW_ABLATE_EXP=1
  arm bapprox_$r TT_BIO_TRIATT_BW_EXP_APPROX=1
  arm sig0_$r    TT_BIO_GATE_SKIP_SIGMOID=1
done
arm base3 X=0
timeout -s INT -k 60 900 "$PY" perf/bc4_elem/prims.py "$D/prims.json" 224 > "$D/prims.log" 2>&1
echo "prims rc=$?" >> "$D/arms"
kill $P; touch "$D/done"
