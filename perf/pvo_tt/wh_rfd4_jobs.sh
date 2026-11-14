#!/bin/bash
# pvo-tt WH RFD4: the PopVax design cell (9TH6 chain A whole, 448 residues + a 120-residue binder = 568 tokens)
# on one lent Galaxy chip, through perf/rfd4/fair/run.py from the MODES pin tree. TT fast = TT exact for RFD4
# (MODES.md, FAST_LEVERS = NORMAL_LEVERS), so there is one arm reported as both modes.
#
#   probe : batch 1, one warm batch. Answers "does 568 tokens run at all on a Wormhole chip", which
#           tt_bio/size_limits.py records for no arch.
#   fair  : the climb plus 5 clean warm batches at the best width, for s per design.
# Both are queued now; the fair job exits 3 (pool requeues it) until the probe has written rc 0.
set -u
R=$HOME/spd/pvo-tt; T=$R/tree-c52ab1702; D=$R/data/rfd4; Q=$HOME/spd/pool/queue
C="TT_BIO_CACHE=$R/cache BOLTZ_CACHE=$R/cache HF_HOME=$R/cache/huggingface"
ln -sf $HOME/spd/spd-orchestrator/run_cell.sh $R/run_cell.sh

# job <name> <prio> <cap_s> <guard line> <extra run.py args>
job() {
  n=$1; p=$2; cap=$3; guard=$4; extra=$5; o=$R/wh-rfd4/$n
  cat > $Q/$p-spd-pvo-tt-rfd4-$n.sh <<JOB
# pvo-tt WH RFD4 $n: 568 tokens (9TH6 chain A + 120 aa binder), thinking_medium 60 steps, tree c52ab1702
# mem: 60
# cap: $cap
o=$o; grep -qsx 0 \$o.rc && exit 0; mkdir -p \$o
$guard
TREE=$T bash $R/run_cell.sh \$CHIP \$o/cell $((cap - 300)) -- bash -c "cd $D && $C python $T/perf/rfd4/fair/run.py --spec $D/9TH6-A-b120.yaml --weights nvidia/RFD4-Proteina --out \$o --part wh --shared-box --max-foreign-cores 60 $extra"
r=\$?; echo \$r > \$o.rc; exit \$r
JOB
}

job probe 3 7200 ':' '--batch 1 --batches 1 --max-attempts 2'
job fair 4 21600 "grep -qsx 0 $R/wh-rfd4/probe.rc || exit 3" '--batches 5 --max-attempts 8'
ls $Q | grep rfd4
