#!/bin/bash
# mol1-fast pLDDT cases on a WH Galaxy pool: the four mol1-whl4 refolds (7fxn 8t65 8fg6 8k83, seed 42, same inputs, cache
# and flags as whl4's gate refolds) per arm, so --fast can be read against upstream fp32 (0.932 / 0.912 / 0.933 / 0.899)
# and the same tree's normal mode. Output: ~/spd/mol1-fast/plddt-<sha>/<arm>/<case>/fold.
#   TREE=~/spd/mol1-fast/tree-<sha> bash plddt_wh.sh [-p PRIO] "n fast dtri ..."
set -u
T=${TREE:?}; S=$(cut -c1-9 $T/TREE_SHA); R=$HOME/spd/mol1-fast/plddt-$S; Q=$HOME/spd/mol1-quality
PRIO=3; [ "${1:-}" = -p ] && { PRIO=$2; shift 2; }
. $T/perf/mol1/fast/arms.sh
mkdir -p $R; ln -sf $HOME/spd/spd-orchestrator/run_cell.sh $R/run_cell.sh
for a in ${1:-n fast}; do arm_env wh $a || exit 1
  for c in 7fxn 8t65 8fg6 8k83; do
    cat > $HOME/spd/pool/queue/$PRIO-spd-mol1-fast-plddt-$S-$a-$c.sh <<JOB
# mol1-fast pLDDT case: $c seed 42, arm $a, tree $S
# mem: 60
# threads: 4
# cap: 2400
export PATH=$Q/kalign/bin:\$PATH TREE=$T TT_BIO_CACHE=$HOME/spd/mol1-port/cache BOLTZ_CACHE=$HOME/spd/mol1-port/cache SHARE=4
${LV:+export TT_BIO_LEVERS=$LV}
d=$R/$a/$c; grep -qs rc=0 \$d/rc && exit 0; mkdir -p \$d
bash $R/run_cell.sh \$CHIP \$d 2400 -- bash -c "cd $Q/inputs && python -c \"from tt_bio.main import cli; cli()\" predict $c-assembly1.json --model mol1 --seed 42 --out_dir \$d/fold $FASTFLAG"
r=\$?; echo rc=\$r > \$d/rc; exit \$r
JOB
    echo $a $c
  done
done
