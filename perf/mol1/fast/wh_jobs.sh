#!/bin/bash
# mol1-fast WH screen: one pool job per (arm, cell) on a Galaxy's lent chips (JapanFold users first: the pool runs only on
# lent chips and yields on recall). perf/mol1/fair/run.py --part wh: load + cold + 3 warm, AICLK sampled in-round.
#   R=~/spd/mol1-fast TREE=<tree dir with TREE_SHA> bash wh_jobs.sh "n f0 b8 fn fb" "9DBP 9W3L"
set -u
R=${R:-$HOME/spd/mol1-fast}; T=${TREE:?}; D=$HOME/spd/pvo-tt/data; S=$(cat $T/TREE_SHA | cut -c1-9)
. $T/perf/mol1/fast/arms.sh
mkdir -p $R; ln -sf $HOME/spd/spd-orchestrator/run_cell.sh $R/run_cell.sh
for a in ${1:-n fb}; do arm_env wh $a || exit 1
  for c in ${2:-9DBP}; do
    j=$HOME/spd/pool/queue/3-spd-mol1-fast-$S-$a-$c.sh; o=$R/wh-$S/$a-$c
    cat > $j <<JOB
# mol1-fast WH screen: arm $a cell $c (5 samples, 20 recycles, 200 steps, frozen MSA cache), tree $S
# mem: 60
# cap: 7200
o=$o; grep -qsx 0 \$o.rc && exit 0; mkdir -p \$o
FW=\$(~/.local/bin/tt-smi -s 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["device_info"][0]["firmwares"]["fw_bundle_version"])' 2>/dev/null)
export TT_BIO_LEVERS=$LV
TREE=$T bash $R/run_cell.sh \$CHIP \$o/cell 6900 -- bash -c "cd $D/inputs && python $T/perf/mol1/fair/run.py --chips \$CHIP --part wh --firmware '\$FW' --model mol1 --arm $a --rounds 3 --max-attempts 4 --input $D/inputs/$c.yaml --out \$o --cli-extra --diffusion_samples 5 --recycling_steps 20 --msa_dir $D/msa/cache --msa_cache_only $FASTFLAG"
r=\$?; echo \$r > \$o.rc; exit \$r
JOB
    echo $j
  done
done
