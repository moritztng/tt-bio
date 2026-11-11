#!/bin/bash
# mol1-fast accuracy grade on a WH Galaxy pool: Mol-1 --fast from this branch's tree on pvo-accuracy's 11 complexes,
# seeds 101-103, same flags and data as pvo-accuracy's Mol-1 exact (perf/pvo_accuracy/wh_jobs.sh on wk/pvo-accuracy),
# so its tt_exact folds are the paired reference. Output: $R/wh/mol1/tt_fast/<PDB>/seed_<s>/.
#   TREE=~/spd/mol1-fast/tree-<sha> bash acc_wh.sh [-p PRIO] [PDB ...]
set -u
T=${TREE:?}; S=$(cut -c1-9 $T/TREE_SHA); R=$HOME/spd/mol1-fast/acc-$S; D=$HOME/spd/pvo-tt/data
PRIO=4; [ "${1:-}" = -p ] && { PRIO=$2; shift 2; }
mkdir -p $R; ln -sf $HOME/spd/spd-orchestrator/run_cell.sh $R/run_cell.sh
for y in $D/inputs/*.yaml; do
  c=$(basename $y .yaml); [ $# -gt 0 ] && [[ " $* " != *" $c "* ]] && continue
  o=$R/wh/mol1/tt_fast/$c
  cat > $HOME/spd/pool/queue/$PRIO-spd-mol1-fast-acc-$S-$c.sh <<JOB
# mol1-fast accuracy: Mol-1 --fast $c seeds 101-103, tree $S, frozen MSA cache
# mem: 60
# cap: 7200
o=$o; mkdir -p \$o
for s in 101 102 103; do
  f=\$o/seed_\$s; grep -qs rc=0 \$f/rc && continue; mkdir -p \$f
  TREE=$T bash $R/run_cell.sh \$CHIP \$f/cell 3000 -- tt-bio predict $y --model mol1 --seed \$s --msa_dir $D/msa/cache --msa_cache_only --out_dir \$f --override --fast
  r=\$?; echo rc=\$r > \$f/rc; [ \$r = 0 ] || exit \$r
done
JOB
  echo $c
done
