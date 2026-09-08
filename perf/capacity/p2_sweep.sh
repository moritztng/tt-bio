#!/bin/bash
# Re-record the Blackhole capacity baseline in batches, so each batch banks its own cells.
# Card 0 on pc, sequential, no reset (a reset takes the only board on this host).
set -u
WT=/home/moritz/.coworker/wt/wh-transition-wchunk-hang-fix-p2
PY=/home/moritz/tt-bio/env/bin/python3
cd $WT
export PYTHONPATH=$WT PYTHONNOUSERSITE=1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:wh-transition-wchunk-hang-fix-p2
for batch in "protenix-v2,openfold3" "rf3,boltz2" "openbind,protenix-v1" \
             "esmc-300m,esmc-600m,saprot-35m,saprot-650m,esmc-6b" \
             "esmfold2,esmfold2-fast,opendde,opendde-abag"; do
  tag=$(echo "$batch" | tr ',' '_' | cut -c1-40)
  echo "===== batch $batch start $(date -u +%FT%TZ)"
  $PY -u scripts/capacity_gate.py --models "$batch" --record --no-card-reset \
      --report $WT/perf/capacity/p2_$tag.json
  echo "===== batch $batch rc=$? end $(date -u +%FT%TZ)"
done
echo "===== sweep done $(date -u +%FT%TZ)"
