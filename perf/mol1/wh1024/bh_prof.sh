#!/bin/bash
# mol1-bh-prof: per-family attribution of Mol-1 on one qb1 p150a at 256/512/1024 (cold, timed warm, attribution warm).
# Usage: bh_prof.sh CARD OUTDIR [SIZES...]   (run detached; holds the card flock for the whole run)
set -u
C=$1 O=$2; shift 2; SIZES=${*:-256 512 1024}
W=$(cd "$(dirname "$0")/../../.." && pwd); PY=/home/ttuser/tt-bio-dev/env/bin/python
mkdir -p $O; say() { echo "$(date -u +%FT%TZ) $*" >> $O/run.log; }
exec 9>/home/ttuser/spd_qb1_card$C.lock; flock -w 1800 9 || { say "card$C lock timeout"; exit 3; }
say "tree $(git -C $W rev-parse HEAD) card $C sizes $SIZES"
cd /home/ttuser/mol1-fair/inputs
for s in $SIZES; do
  uptime >> $O/run.log
  TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mol1-bh-prof PYTHONPATH=$W timeout 7200 \
    $PY $W/perf/mol1/wh1024/prof.py --input mol1-$s.json --chip $C --out $O/p$s > $O/p$s.log 2>&1
  say "size $s rc=$?"
done
say done; touch $O/DONE
