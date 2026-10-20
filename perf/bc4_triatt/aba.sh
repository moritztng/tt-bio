#!/bin/bash
# bc4-bet-triatt step 3: model-level A/B/A of the BindCraft 2 iteration body on one qb2 chip, K=48 (all 48
# Evoformer tapes resident, main's setting on a p150a), A = shipped levers, B = A + $ARM (env assignments).
# Legs A,B,A,B,A,B,A interleaved, `bc2_evo_devprof.py --warm 2 --reps 6`, AICLK + load at 1 Hz.
#   ARM="TT_BIO_TRIATT_BW_EXP_APPROX=1 TT_BIO_TRIATT_BW_V2=1" setsid nohup bash perf/bc4_triatt/aba.sh &
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=${OUT:-$WT/_bc4out/aba_$(git rev-parse --short HEAD)}; mkdir -p "$D"
PY=$HOME/bcx_e2e_venv/bin/python
PARAMS=$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz
if [ -z "${CARD:-}" ]; then
  echo $$ > "$D/pid"; git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"
  echo "${ARM:?set ARM}" > "$D/arm"
  for c in ${CHIPS:-3 2 1 0}; do
    ( flock 9 && mkdir "$D/claim" 2>/dev/null && CARD=$c exec bash "$0" ) 9>"$HOME/spd_qb2_chip$c.lock" &
  done
  wait; exit 0
fi
ARM=$(cat "$D/arm")
echo "card $CARD start $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> "$D/card"
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-triatt
export TT_BIO_BC2_RESIDENT_BLOCKS=48
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk 2>/dev/null)" \
    "$(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
leg() { local n=$1; shift; echo "$n start $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> "$D/card"
  env "$@" timeout -s INT -k 120 2400 $PY perf/spd/bc2_evo_devprof.py --params "$PARAMS" --out "$D/$n" \
    --warm 2 --reps 6 > "$D/$n.log" 2>&1
  echo "$n rc=$? end $(date +%s)" >> "$D/card"; }
leg A0 X=0; leg B0 $ARM; leg A1 X=0; leg B1 $ARM; leg A2 X=0; leg B2 $ARM; leg A3 X=0
env $ARM timeout -s INT -k 120 3000 $PY perf/spd/bc2_evo_devprof.py --params "$PARAMS" --out "$D/Bcen" --warm 2 --reps 2 --census > "$D/Bcen.log" 2>&1
echo "Bcen rc=$? end $(date +%s)" >> "$D/card"
kill $P; touch "$D/done"
