#!/bin/bash
# Queue one SPD-pool job per target on a held Galaxy (state/spd/CHIPS.md, "Chip pool"): the pool starts it as
# `CHIP=<umd> bash <job>` on an idle healthy chip. Run ON the box.
# usage: queue.sh TREE SET TARGET... [PRIO=8] [EXTRA="--seeds 101"]
#   TREE = dir under ~/spd/tfg-accuracy/trees (a git archive with a SHA file)
#   SET  = dir under ~/spd/tfg-accuracy/data (panel layout: <tid>/<tid>_<cond>.json + MSAs)
# Output: ~/spd/tfg-accuracy/runs/<SET>/<tid>/ (perf/tfg_acc/run.py layout), log run.log there.
set -eu
TREE=$1; SET=$2; shift 2
R=$HOME/spd/tfg-accuracy
for T in "$@"; do
  job=$HOME/spd/pool/queue/${PRIO:-8}-tfg-accuracy-$SET-$T.sh
  cat > "$job" <<EOF
# mem: 30
. ~/japanfold/env.sh > /dev/null 2>&1
cd $R/trees/$TREE
export PYTHONPATH=\$PWD TT_BIO_LEASE_HOLDER=tfg-accuracy TT_BIO_LEASE_DIR=$R/leases \\
       TT_METAL_CACHE=$R/cache-$TREE-\$CHIP OMP_NUM_THREADS=2
mkdir -p $R/leases $R/runs/$SET/$T
exec 9> ~/spd/locks/chip\$CHIP.lock
flock -w 600 9 || { echo "\$(date -u +%FT%TZ) flock busy chip \$CHIP" >> $R/runs/$SET/$T/run.log; exit 3; }
echo "\$(date -u +%FT%TZ) start chip \$CHIP tree $TREE" >> $R/runs/$SET/$T/run.log
timeout -s TERM 43260 timeout -s INT 43200 nice -n 10 python perf/tfg_acc/run.py --panel $R/data/$SET \\
  --target $T --out $R/runs/$SET/$T --chip \$CHIP --share 32 ${EXTRA:-} >> $R/runs/$SET/$T/run.log 2>&1
echo "\$(date -u +%FT%TZ) end rc=\$?" >> $R/runs/$SET/$T/run.log
EOF
  echo "queued $job"
done
