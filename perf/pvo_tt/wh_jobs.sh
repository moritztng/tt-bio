#!/bin/bash
# pvo-tt WH: one pool job per (model, mode, complex) on .107's lent chips. Each runs perf/mol1/fair/run.py --part wh
# (load + cold + 3 warm on one chip, AICLK sampled in-round) from the MODES pin tree. Restartable: a done cell is skipped.
set -u
R=$HOME/spd/pvo-tt; T=$R/tree-c52ab1702; D=$R/data
echo c52ab1702031 > $T/TREE_SHA
ln -sf $HOME/spd/spd-orchestrator/run_cell.sh $R/run_cell.sh
for arm in opendde:exact opendde:fast mol1:exact; do
  m=${arm%%:*}; mode=${arm#*:}; X=""; [ $mode = fast ] && X=--fast; rc=10; [ $m = mol1 ] && rc=20
  for c in 9TH6 9DBP 9LV4 9W3L; do
    j=$HOME/spd/pool/queue/3-spd-pvo-tt-$m-$mode-$c.sh; o=$R/wh/$m-$mode-$c
    cat > $j <<JOB
# pvo-tt WH timing: $m $mode $c (5 samples, $rc recycles, 200 steps, frozen MSA cache), tree c52ab1702 (state/pvo/MODES.md pin)
# mem: 60
# cap: 7200
o=$o; grep -qsx 0 \$o.rc && exit 0; mkdir -p \$o
FW=\$(~/.local/bin/tt-smi -s 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["device_info"][0]["firmwares"]["fw_bundle_version"])' 2>/dev/null)
TREE=$T bash $R/run_cell.sh \$CHIP \$o/cell 6900 -- bash -c "cd $D/inputs && python $T/perf/mol1/fair/run.py --chips \$CHIP --part wh --firmware '\$FW' --model $m --arm $mode --rounds 3 --max-attempts 4 --input $D/inputs/$c.yaml --out \$o --cli-extra --diffusion_samples 5 --recycling_steps $rc --msa_dir $D/msa/cache --msa_cache_only $X"
r=\$?; echo \$r > \$o.rc; exit \$r
JOB
  done
done
ls $HOME/spd/pool/queue | grep pvo-tt
