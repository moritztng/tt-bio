#!/bin/bash
# qb2 (pc chip out of service 10-24): 192 at auto_width (14) on the merge base and on the branch, then
# drain_parity at widths 2, 4, 8, 32. Books card N through the qb2 arbiter. Results in $R.
N=${1:-1} R=${2:-$HOME/scratch/l1d-runs}; HERE=$(cd "$(dirname "$0")" && pwd); WT=$(cd "$HERE/../../.." && pwd)
BOOK="python3 $HOME/spd/feed/book.py" OWN=rfd4-l1decline
export PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python} IN=$HOME/rfd4-inputs W=$HOME/rfd4-bundle CARD=$N
mkdir -p $R/main; [ -e $R/main/tt_bio ] || git -C $WT archive 77ca47583 | tar -x -C $R/main
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N 3600 > /dev/null; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { echo "flock timeout" > $R/chain4.done; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 15; done
echo "start card $N $(date -u +%FT%TZ)" > $R/chain4.log
$HERE/repro.sh $R/main-192b14 192 14 $R/main
$HERE/repro.sh $R/branch-192b14 192 14
mkdir -p /tmp/l1d-empty && cd /tmp/l1d-empty && PYTHONPATH=$WT TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N \
  TT_BIO_LEASE_HOLDER=worker:$OWN timeout 1500 $PY $HERE/drain_parity.py 2 4 8 32 > $R/drain_parity.log 2>&1
echo "rc=$? $(date -u +%FT%TZ)" >> $R/drain_parity.log
$BOOK rm $OWN
echo "chain4 done $(date -u +%FT%TZ)" > $R/chain4.done
