#!/bin/bash
# bc4-bet-mm2: full-iteration A/B/A of the split-K rule on one qb2 chip, detached.
#   CHIP=2 WARM=2 setsid nohup bash perf/bc4_mm2/aba.sh <name> > /dev/null 2>&1 &     marker _mm2/<name>/done
set -u
NAME=$1; C=${CHIP:?}; WARM=${WARM:-2}
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT" || exit 1
D=$WT/_mm2/$NAME; mkdir -p "$D"
git rev-parse HEAD > "$D/sha"; git status --porcelain -uno > "$D/dirty"
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bc4-bet-mm2 TT_BIO_LEASE_TIMEOUT=7200
export JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params
export JAPANFOLD_BC2_MPNN_WEIGHTS=$HOME/bcx_e2e/bc2/bindcraft/weights/proteinmpnn/weights_neutral
unset LD_LIBRARY_PATH
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$C/tt_aiclk)" \
    "$(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" & CLKP=$!
trap 'kill $CLKP 2>/dev/null' EXIT
for arm in A:0 B:1 A2:0; do
  n=${arm%%:*}; v=${arm#*:}
  echo "$(date -u +%FT%TZ) $n TT_BIO_MM_SPLIT_K=$v start load $(cut -d' ' -f1 /proc/loadavg)" >> "$D/chain.log"
  TT_BIO_MM_SPLIT_K=$v flock -w 600 "$HOME/spd_qb2_chip$C.lock" timeout 5400 "$HOME/bcx_e2e_venv/bin/python" \
      perf/spd/bench.py --out "$D/$n" --chip "$C" --model bindcraft2 --bc2 "$HOME/bcx_e2e/bc2" --seed 101 \
      --warm "$WARM" --inputs 80 > "$D/$n.log" 2>&1
  echo "$(date -u +%FT%TZ) $n rc=$?" >> "$D/chain.log"
done
echo "done $(date -u +%FT%TZ)" > "$D/done"
