#!/bin/bash
# Precondition: is OpenFold3 confident on ONE CDK2 monomer when it is given the MSA?
#
# Every fold this row has taken was --single_sequence, which is the obvious candidate reason the
# confidence heads read pLDDT 0.37 at 832. If the monomer with its own alignment is confident,
# the tiled fixture is usable and the 832 arms are worth taking. If it is not, this whole fixture
# family is the wrong one and no amount of tiling fixes it.
#
# --msa_cache_only so nothing can silently reach the network and change the alignment.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/msafix
CARD=3
cd "$WT" || exit 1

run () {
  tag=$1
  n=$2
  t0=$(date +%s.%N)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model openfold3 \
        --msa_dir "$OUT/msa" --msa_cache_only \
        --sampling_steps 20 --diffusion_samples 1 \
        --out_dir "$OUT/res_$tag" "$OUT/cdk2_msa_$n.yaml" > "$OUT/log_$tag.txt" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  echo "LEG $tag n=$n rc=$rc wall=$(echo "$t1 - $t0" | bc)s"
  cat "$OUT/res_$tag/openfold3_results_cdk2_msa_$n/results.json" 2>/dev/null | tr -d '\n '
  echo
}

run msa298 298
echo PRECOND_DONE
