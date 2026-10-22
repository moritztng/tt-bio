#!/bin/bash
# rfd4-perf-hostlane device leg, ONE qb2 chip under its spd lock. Trees are `git archive` exports:
#   $S/base  = the branch before the conditioning change (d3b7d6c65: main + pairdev merge + host_step.py)
#   $S/after = the change (pass the sha as $2)
# Legs, interleaved base/after: host_split (glue split with the card running) at 512 and 256, the after
# tree again at OMP_NUM_THREADS=4 (is host glue slowed by thread contention with ttnn?), then design.py
# for wall s/design and atoms identity (same seed, rng upstream) per size.
set -u
C=$1 AFTER=$2
S=$HOME/scratch/hostlane-dev; R=$S/out
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
W=$HOME/rfd4-bundle
mkdir -p $R
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
exec 9>$HOME/spd_qb2_chip$C.lock
stamp "wait flock chip$C"
flock -w 10800 9 || { stamp "no lock on chip $C in 3 h"; exit 3; }
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-hostlane TT_BIO_TRACE_REGION_MB=1536
stamp "start chip $C base $(cat $S/base/.sha) after $(cat $S/after/.sha)"
split() {  # tree size tag [env...]
  local t=$1 n=$2 tag=$3; shift 3
  (cd $S/$t && env "$@" PYTHONPATH=$S/$t timeout 3600 $PY perf/rfd4/e2e/host_split.py --spec $HOME/rfd4-inputs/pdl1-$n.yaml \
    --weights $W --out $R/split$n --designs 3 --tag $tag-) > $R/split$n-$tag.log 2>&1
  stamp "split $n $tag rc=$?"
}
design() {  # tree size tag
  local t=$1 n=$2 tag=$3
  (cd $S/$t && PYTHONPATH=$S/$t timeout 3600 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-$n.yaml \
    --weights $W --out $R/d$n-$tag --rng upstream --seed 0 --designs 3) > $R/d$n-$tag.log 2>&1
  stamp "design $n $tag rc=$?"
}
for n in 512 256; do
  split base $n base X=1
  split after $n after X=1
  split after $n omp4 OMP_NUM_THREADS=4
  design base $n base1
  design after $n after1
  design base $n base2
  design after $n after2
  for f in $(cd $R/d$n-base1 && ls *.cif); do
    $PY $S/after/perf/rfd4_land/cif_compare.py $R/d$n-base1/$f $R/d$n-after1/$f --json $R/id$n-$f.json >> $R/identity.txt 2>&1
    stamp "identity $n $f rc=$?"
  done
done
stamp ALLDONE
