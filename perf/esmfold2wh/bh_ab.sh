#!/bin/bash
# Blackhole neutrality for the Wormhole MSA LM-release fix. is_wormhole() is False on a
# p150a, so release_lm can never be True and the dram_peak tags are no-ops with
# TT_BIO_DRAM_PEAK unset. The proof is a byte-identical CIF, not the argument.
WT=/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-unusable
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=$WT/perf/esmfold2wh
cd $WT || exit 1
leg() {
  TT_VISIBLE_DEVICES=2 TT_METAL_LOGGER_LEVEL=FATAL   TT_BIO_LEASE_HOLDER=worker:japanfold-esmfold2-wh-unusable   PYTHONPATH=$WT timeout 2400 $PY -m tt_bio.main predict $O/cdk2_512.fasta     --model esmfold2 --fast --single_sequence --seed 0     --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1     --out_dir $O/out_$1 --override > $O/bh_$1.log 2>&1
  echo "EXIT $1 = $?" >> $O/bh_$1.log
}
leg fixed
git checkout HEAD~1 -- tt_bio/esmfold2_runtime.py
leg base
git checkout HEAD -- tt_bio/esmfold2_runtime.py
sha256sum $O/out_fixed/*/structures/*.cif $O/out_base/*/structures/*.cif > $O/bh_ab_sha.txt 2>&1
echo DONE > $O/.bh_done
