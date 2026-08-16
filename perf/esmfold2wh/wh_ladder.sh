#!/bin/bash
# The Wormhole ladder for the MSA-encoder row blocking. Run on UF-EV-A13-GWH02,
# UMD 28 / node 4 (the one healthy spare outside the JapanFold pool).
# Takes the device lease; if another worker's chain holds it this exits rather
# than fighting for it. 640 is the regression control: it folded in 230.4 s
# before this change and must still fold, byte-identically if nothing moved.
#
#   ./wh_ladder.sh                 -> 640, 788, 896, 1024 at the default depth
#   ./wh_ladder.sh depth 1024 4096 -> the max_msa_seqs probe at one length
set -u
# SRC must first be brought to this branch: the GWH02 tree sits at 490da2c0 plus
# the predecessor's fix, and does NOT carry 84c335ba/a22e2ec2 yet.
SRC=${SRC:-/home/cust-team/mthuening/esmfold2wh/src}
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3
D=/home/cust-team/mthuening/esmfold2wh
run() {  # $1=tag $2=fasta $3=extra args
  env -i HOME=/home/cust-team PATH=/usr/bin:/bin \
    HF_HUB_CACHE=/home/cust-team/models \
    TT_VISIBLE_DEVICES=28 \
    TT_BIO_LEASE_HOLDER=worker:japanfold-esmfold2-wh-msa-cap-p2 \
    TT_BIO_DRAM_PEAK=$D/dram_$1.log \
    PYTHONPATH=$SRC \
    timeout 3000 $PY -m tt_bio.main predict $D/$2 \
      --model esmfold2 --fast --use_msa_server --msa_dir /data/msa_cache --seed 0 \
      --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 \
      $3 --out_dir $D/out_$1 --override > $D/run_$1.log 2>&1
  echo "$1 exit=$? $(ls $D/out_$1/*/structures/*.cif 2>/dev/null | wc -l) cif"
}
if [ "${1:-}" = "depth" ]; then
  run "p2_${2}_m$3" "cdk2_$2.fasta" "--max_msa_seqs $3"
  exit
fi
for L in 640 788 980 1024; do
  [ -f "$D/cdk2_$L.fasta" ] || { echo "skip $L (no fixture)"; continue; }
  run "p2_$L" "cdk2_$L.fasta" ""
done
