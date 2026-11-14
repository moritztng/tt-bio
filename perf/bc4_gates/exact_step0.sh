#!/bin/bash
# bc4-bet-gates step 0: is the per-channel trimul default exact end to end on Blackhole? One fold per model, input
# and arm through perf/spd/bench.py (same seeds), per-channel off (TT_BIO_TRIMUL_PER_CHANNEL=0) vs the default;
# the coords digests must match. Runs a snapshot of HEAD behind the card's fleet lock.
#   CARD=1 bash perf/bc4_gates/exact_step0.sh
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
CARD=${CARD:?CARD=<leased chip>}
BR=$(git rev-parse --short=9 HEAD); D=/home/moritz/.coworker/state/bc4/bets/gates/raw/exact_$BR; mkdir -p $D
S=$WT/_bc4out/src_$BR; rm -rf $S; mkdir -p $S; git archive $BR | tar -x -C $S; cd $S
PY=$HOME/bcx_e2e_venv/bin/python
export PYTHONPATH=$S TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bc4-bet-gates}
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
exec 9>"$HOME/spd_qb2_chip$CARD.lock"; flock 9
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
echo "start $(date +%s) sha $BR card $CARD snapshot=$S load $(cut -d" " -f1 /proc/loadavg)" > $D/card
for m in boltz2 protenix-v2 openfold3; do for arm in off:TT_BIO_TRIMUL_PER_CHANNEL=0 on; do
  n=${m}_${arm%%:*}; echo "$n start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> $D/card
  timeout -s INT -k 120 2400 $PY perf/spd/bench.py --out $D/$n --chip $CARD --model $m --arm $arm \
      --inputs l224,l256 --warm 1 --samples 1 --no-coords > $D/$n.log 2>&1
  echo "$n rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> $D/card
done; done
touch $D/done
