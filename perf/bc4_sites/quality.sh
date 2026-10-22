#!/bin/bash
# bc4-bet-sites quality arm on one qb2 chip: BindCraft 2 PD-L1, --warm 9 --inputs 80, one bench.py per seed,
# from the tree this script sits in.   quality.sh <chip> <seed>...   Out: _bc4out/q/tt_s<seed>, `done.<chip>`.
set -u
C=$1; shift
T=$(cd "$(dirname "$0")/../.." && pwd); D=$T/_bc4out/q; mkdir -p "$D"; cd "$T"
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bc4-bet-sites
export JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params JAPANFOLD_BC2_MPNN_WEIGHTS=$HOME/bcx_e2e/bc2/bindcraft/weights/proteinmpnn/weights_neutral
export PYTHONPATH=$T SPD_TREE_SHA=$(cat sha)
PY=$HOME/bcx_e2e_venv/bin/python3; LOCK=$HOME/spd_qb2_chip$C.lock
( while :; do echo "$(date +%s) $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk) $(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > "$D/aiclk.$C.tsv" &
SAMPLER=$!
for S in "$@"; do
  echo "s$S start $(date +%s) chip $C" >> "$D/card"
  flock "$LOCK" timeout -k 60 14000 $PY perf/spd/bench.py --out "$D/tt_s$S" --chip $C --model bindcraft2 \
      --bc2 $HOME/bcx_e2e/bc2 --seed $S --warm 9 --inputs 80 > "$D/s$S.log" 2>&1
  echo "s$S rc=$? $(date +%s)" >> "$D/card"
done
kill $SAMPLER; touch "$D/done.$C"
