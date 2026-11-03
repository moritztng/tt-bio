#!/bin/bash
# Run ON a WH Galaxy (.107) that lends chips to its SPD pool: one pool job per (model, mode, complex), each folding
# seeds 101-103 on whichever lent chip pool.sh hands it, through the shared run_cell.sh (chip flock, private lease,
# AICLK of every node at 2 Hz in <cell>/aiclk.tsv). Reuses pvo-tt's pinned tree (c52ab1702, the MODES.md pin) and its
# data copy (inputs sha256-identical to pc's ~/pvo-accuracy-data/inputs, same MSA cache), so nothing is re-staged.
# Same flags as run_tt.sh: model-default recycles/steps, opendde --diffusion_samples 5, Mol-1 exact only (MODES.md).
# Priority 4, behind pvo-tt's timing jobs; queued complex by complex so a short window still leaves exact/fast pairs. Resumable: a fold with rc=0 is skipped, a job whose 3 folds passed exits 0.
# Output layout = run_tt.sh's ($R/wh/<model>/tt_<mode>/<PDB>/seed_<s>/), so tt_relayout.py reads it after the pull.
set -u
R=$HOME/spd/pvo-accuracy; P=$HOME/spd/pvo-tt; T=$P/tree-c52ab1702; D=$P/data
mkdir -p $R/wh; ln -sf $HOME/spd/spd-orchestrator/run_cell.sh $R/run_cell.sh
for y in $D/inputs/*.yaml; do
  c=$(basename $y .yaml)
  for arm in opendde:exact opendde:fast mol1:exact; do
    m=${arm%%:*}; mode=${arm#*:}; X=""; [ $mode = fast ] && X=--fast; S=""; [ $m = opendde ] && S="--diffusion_samples 5"
    o=$R/wh/$m/tt_$mode/$c
    cat > $HOME/spd/pool/queue/4-spd-pvo-accuracy-$m-$mode-$c.sh <<JOB
# pvo-accuracy WH: $m $mode $c seeds 101-103 (model-default recycles/steps $S), tree c52ab1702, frozen MSA cache
# mem: 60
# cap: 7200
o=$o; mkdir -p \$o
for s in 101 102 103; do
  f=\$o/seed_\$s; grep -qs rc=0 \$f/rc && continue; mkdir -p \$f
  TREE=$T bash $R/run_cell.sh \$CHIP \$f/cell 3000 -- tt-bio predict $y --model $m --seed \$s --msa_dir $D/msa/cache --msa_cache_only --out_dir \$f --override $S $X
  r=\$?; echo rc=\$r > \$f/rc; [ \$r = 0 ] || exit \$r
done
JOB
    sleep 1
  done
done
ls $HOME/spd/pool/queue | grep -c pvo-accuracy
