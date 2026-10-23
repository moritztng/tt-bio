#!/bin/bash
# The fair harness at 192 / 256 / 320 on one qb2 chip (dev line, p300c, --shared-box): a batch as one lane at the auto
# width against the same lane width with two or four lanes in the batch, so one lane`s host work hides behind the
# other`s trunk. Runs after run.sh on the same booking; waits for its ALLDONE.
#   bash perf/rfd4/p256/fairlanes.sh <chip> <outdir> <run.sh outdir>
set -u
N=$1; R=$2; R1=$3; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-256
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
until grep -q ALLDONE $R1/done 2>/dev/null; do sleep 30; done
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-3600} --now --note "RFD4 256 fair lanes A/B")"
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 1800 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(git rev-parse --short HEAD)"
for arm in 256:4:4 256:8:4 256:16:4 256:4:4b 192:7:7 192:14:7 320:2:2 320:4:2; do
  IFS=: read n b lw <<< "$arm"; lw=${lw%b}
  timeout 1800 $PY perf/rfd4/fair/run.py --spec $IN/pdl1-$n.yaml --weights $W --out $R/${arm//:/-} --part p300c \
    --shared-box --max-foreign-cores 99 --quiet-timeout 0 --batch $b --lane-width $lw --batches 3 > $R/${arm//:/-}.log 2>&1
  stamp "$arm rc=$?"
done
exec 9>&-; $BOOK rm $OWN; stamp ALLDONE
