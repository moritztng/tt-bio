#!/bin/bash
# rfd4-perf-tri224 run6: other models digests of the two shared kernels (digest.py), main vs this branch, on a chip
# without a determinism marker. The levers themselves are RFD4-keyed; this checks the kernel sources they edited.
#   bash perf/rfd4/tri224/run6.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
BOOK="python3 $HOME/spd/feed/book.py"; OWN=${OWN:-rfd4-perf-tri224}
mkdir -p $R; R=$(cd $R && pwd)
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
[ -e $HOME/spd/locks/chip$N.BAD ] && { stamp "chip $N carries chip$N.BAD, refusing"; exit 2; }
MAIN=$R/main-tree; rm -rf $MAIN; mkdir -p $MAIN
git -C ${SRC:-$WT} archive ${MAIN_REF:-origin/main} | tar -x -C $MAIN
# digest.py is new on this branch: main runs the same script against its own tt_bio
mkdir -p $MAIN/perf/rfd4/tri224; cp $WT/perf/rfd4/tri224/digest.py $MAIN/perf/rfd4/tri224/
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-1200} --note "tri224 run6 digests" > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N main ${MAIN_REF:-origin/main} tree $(basename $WT)"
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
for t in main branch; do D=$WT; [ $t = main ] && D=$MAIN
  (cd $D && PYTHONPATH=$D timeout 1500 $PY perf/rfd4/tri224/digest.py --n 256 1024 --out $R/$t.json > $R/$t.log 2>&1)
  stamp "$t rc=$?"
done
python3 -c "import json,sys; a,b=(json.load(open(\"$R/\"+t+\".json\")) for t in (\"main\",\"branch\")); print(\"EQUAL\" if a==b else \"DIFF \"+str({k:(a.get(k),b.get(k)) for k in set(a)|set(b) if a.get(k)!=b.get(k)}))" >> $R/done 2>&1
exec 9>&-; $BOOK rm $OWN > /dev/null 2>&1
stamp ALLDONE
