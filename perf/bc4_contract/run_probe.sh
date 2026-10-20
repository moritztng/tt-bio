#!/bin/bash
# Detached runner for the bc4-bet-contract probe: waits for the card to have no open fd and its fleet lock, samples
# AICLK and load at 1 Hz beside it.   CARD=2 bash perf/bc4_contract/run_probe.sh
# Cells: N=224, C=128 (Evoformer, extra-MSA) and C=64 (template), operands in DRAM as the taped BC2 path holds them,
# each deferred transpose the model hands the contraction (a: ending, b: starting, none: the VJP's plain form).
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
CARD=${CARD:?CARD=<leased chip>}
D=$WT/_bc4out/contract_$(git rev-parse --short HEAD); mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-contract
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
exec 9>"$HOME/spd_qb2_chip$CARD.lock"; flock 9
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"; echo "$CARD" > "$D/cardno"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
run() { local tag=$1; shift; echo "$tag start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
  timeout -s INT -k 60 900 "$PY" perf/bc4_contract/probe.py "$@" > "$D/$tag.log" 2>&1
  echo "$tag rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"; }
run control control "$D/control.json"
for m in dram l1; do for t in a b none; do
  run "sweep_${m}_$t" sweep --out "$D/sweep_${m}_$t.json" --sizes $([ $m = l1 ] && echo 224,160,192 || echo 224,160,192,256,288,320) --channels 128,64 --mem $m --transpose $t \
      --no-reuse2 --res --inner 10 --reps 20; done
done
run control_end control "$D/control_end.json"
kill $P; touch "$D/done"
