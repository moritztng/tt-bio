#!/bin/bash
# Blackhole neutrality for the MSA-encoder row blocking, on an MSA-BEARING fold.
# The predecessor's bh_ab.sh used --single_sequence, which never enters this code.
# On a 110-core grid SMALL_GRID_MSA_TILE_AREA is 0 and pair_row_tile returns 0, so
# every block loop degenerates to the shipped call. The byte-identical CIF is the proof.
WT=/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-msa-cap-p2
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=/home/ttuser/esmfold2wh_msa/fixtures
CACHE=/home/ttuser/esmfold2wh_msa/cache
O=$WT/perf/esmfold2wh
BASE=23411438
cd $WT || exit 1
leg() {
  TT_VISIBLE_DEVICES=1 TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:japanfold-esmfold2-wh-msa-cap-p2 \
  PYTHONPATH=$WT timeout 3000 $PY -m tt_bio.main predict $FIX/cdk2_512.fasta \
    --model esmfold2 --fast --use_msa_server --msa_dir $CACHE --seed 0 \
    --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 \
    --out_dir $O/out_msa_$1 --override > $O/bh_msa_$1.log 2>&1
  echo "EXIT $1 = $?" >> $O/bh_msa_$1.log
}
leg fixed
git checkout $BASE -- tt_bio/esmfold2.py tt_bio/tenstorrent.py
leg base
git checkout HEAD -- tt_bio/esmfold2.py tt_bio/tenstorrent.py
sha256sum $O/out_msa_fixed/*/structures/*.cif $O/out_msa_base/*/structures/*.cif > $O/bh_msa_ab_sha.txt 2>&1
cat $O/bh_msa_ab_sha.txt
