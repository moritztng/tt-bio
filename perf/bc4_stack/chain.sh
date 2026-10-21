#!/bin/bash
# bc4-bet-stack: grade the stack branch (StructureOnDevice + fast trimul sigmoid, both defaults) on designs, as qmain grades main.
#   CARD=<chip> SEEDS="101 102" [WAITPID=<pid>] setsid nohup bash perf/bc4_stack/chain.sh > /dev/null 2>&1 &
# One chain per chip; output lands flat in _bc4out/s_<sha>/ so grade.sh can rsync the dir as the arm.
set -u
CARD=${CARD:?}
SEEDS=${SEEDS:?}
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
SHA=$(git rev-parse --short HEAD); D=${QDIR:-$WT/_bc4out/s_$SHA}; mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-stack TT_BIO_LEASE_TIMEOUT=7200
export JAPANFOLD_BC2_AF2_WEIGHTS=${JAPANFOLD_BC2_AF2_WEIGHTS:-$HOME/bcx_e2e/af2_params}
export JAPANFOLD_BC2_MPNN_WEIGHTS=${JAPANFOLD_BC2_MPNN_WEIGHTS:-$HOME/bcx_e2e/bc2/bindcraft/weights/proteinmpnn/weights_neutral}
L="$D/chain_c$CARD.log"
log() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }

if [[ -n "${WAITPID:-}" ]]; then
  log "waiting for pid $WAITPID to exit before taking chip $CARD"
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
  log "pid $WAITPID gone"
fi

( while :; do printf "%s\t%s\t%s\t%s\n" "$(date +%s)" "$CARD" \
    "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk_c$CARD.tsv" & CLKP=$!
trap 'kill $CLKP 2>/dev/null' EXIT

echo "chip $CARD seeds $SEEDS sha $(git rev-parse HEAD) dirty $(git status --porcelain -uno | wc -l)" >> "$D/card"
log "start pid $$ chip $CARD seeds [$SEEDS] sha $(git rev-parse HEAD) dirty $(git status --porcelain -uno | wc -l)"
for s in $SEEDS; do
  log "s$s start load $(cut -d' ' -f1 /proc/loadavg) aiclk $(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)"
  timeout 28800 "$PY" perf/spd/bench.py --out "$D/tt_s$s" --chip "$CARD" --model bindcraft2 \
      --bc2 "$HOME/bcx_e2e/bc2" --seed "$s" --warm 9 --inputs 80 > "$D/tt_s$s.log" 2>&1
  log "s$s rc=$? crashes $(grep -ci crash "$D/tt_s$s.log" 2>/dev/null)"
done
touch "$D/done_c$CARD"; log "done chip $CARD"
