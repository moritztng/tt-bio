#!/bin/bash
# pvo-tt WH RFD4 design arm for pvo-accuracy (state/pvo/RFD4-ACC.md): 128 designs at the PVO target, split
# into 8 shards of 16 so each takes one lent chip for ~50 min instead of one chip for 6 h. Seeds 1234..1361,
# the GPU side's block (perf/pvo_gpu/pvo_rfd4.py DESIGN_SEED = 1234), 16 consecutive seeds a shard.
#
# The target is the GPU side's frozen 9TH6-A-b120 (antigen whole, 448 res + a 120 aa binder = 568 tokens), NOT
# INPUTS.md's crop: the 12 A crop around the epitope leaves 346 residues in 30 segments, which is not a target
# anyone would run, so both sides design against the whole antigen (state/pvo-tt.md pass 1).
#
# Batch 1 because that is RFD4's own-best width at this size on a Wormhole chip, measured: batch 2 and batch 4
# both run ~249 s a design against batch 1's 168 s (state/pvo/tt/wh-rfd4/fair.jsonl).
set -u
R=$HOME/spd/pvo-tt; T=$R/tree-c52ab1702; D=$R/data/rfd4; Q=$HOME/spd/pool/queue; H=$R/harness
C="TT_BIO_CACHE=$R/cache BOLTZ_CACHE=$R/cache HF_HOME=$R/cache/huggingface PYTHONPATH=$T:$T/perf"
ln -sf $HOME/spd/spd-orchestrator/run_cell.sh $R/run_cell.sh

N=16
for k in 0 1 2 3 4 5 6 7; do
  s0=$((1234 + k * N)); o=$R/wh-rfd4-acc/sh$k
  cat > $Q/4-spd-pvo-tt-rfd4-acc-sh$k.sh <<JOB
# pvo-tt WH RFD4 designs shard $k: seeds $s0..$((s0 + N - 1)), 568 tokens, batch 1, tree c52ab1702
# mem: 60
# cap: 7200
o=$o; grep -qsx 0 \$o.rc && exit 0; mkdir -p \$o
grep -qsx 0 $R/wh-rfd4/probe.rc || exit 3
TREE=$T bash $R/run_cell.sh \$CHIP \$o/cell 6900 -- bash -c "cd $D && $C python $H/rfd4_designs.py --spec $D/9TH6-A-b120.yaml --weights nvidia/RFD4-Proteina --out \$o --n $N --seed0 $s0"
r=\$?; echo \$r > \$o.rc; exit \$r
JOB
done
ls $Q | grep rfd4-acc
