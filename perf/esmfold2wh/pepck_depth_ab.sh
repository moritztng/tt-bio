#!/bin/bash
# Does the Wormhole MSA depth cap move DOMAINS?
#
# cdk2_640 could not answer this: it is a tiled chimera joined by a floppy linker, so a
# global superposition is saturated by hinge motion and a half split cuts through a domain
# (state/japanfold-esmfold2-wh-msa-cap-p2.md S21). This runs the same one-flag-apart A/B on
# a NATURAL multi-domain single chain with an experimental structure to score against:
# human cytosolic PEPCK, PDB 1KHB chain A, 625 aa SEQRES, 1.85 A, two domains.
#
# Both arms are Blackhole (qb1 card 2). The cap itself is Wormhole-gated, but what it does
# to a structure is truncate the alignment from 8192 rows to 5120 -- an input change, not an
# arch one -- and the fleet's only Wormhole part is the live JapanFold Galaxy, held by
# japanfold-prod-sync-esmfold2-levers' deploy window at write time.
set -u
WT=/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-msa-cap-p3-prove
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=/home/ttuser/msacap_p3
cd "$WT" || exit 1
arm() {  # $1 = tag, $2 = --max_msa_seqs
  TT_VISIBLE_DEVICES=2 TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:japanfold-esmfold2-wh-msa-cap-p3-prove \
  PYTHONPATH=$WT timeout 4000 $PY -m tt_bio.main predict $D/pepck_1khb.fasta \
    --model esmfold2 --fast --msa_dir $D/msa --seed 0 --max_msa_seqs "$2" \
    --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 \
    --out_dir "$D/out/$1" --override > "$D/run_$1.log" 2>&1
  echo "$1 exit=$? cif=$(ls $D/out/$1/*/structures/*.cif 2>/dev/null | wc -l)"
}
arm full 8192
arm cap5120 5120
sha256sum $D/out/full/*/structures/*.cif $D/out/cap5120/*/structures/*.cif
