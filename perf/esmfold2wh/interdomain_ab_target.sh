#!/bin/bash
# The 8192-vs-5120 A/B for one target. $1 = tag, $2 = fasta, $3 = experimental cif.
# The MSA must already be cached in $D/msa (interdomain_ab.sh has the fetch-and-retry loop).
set -u
WT=/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-msa-cap-p3-prove
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=/home/ttuser/msacap_p3
TAG=$1; FASTA=$2; EXP=$3
cd "$WT" || exit 1
arm() {
  TT_VISIBLE_DEVICES=2 TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:japanfold-esmfold2-wh-msa-cap-p3-prove \
  PYTHONPATH=$WT timeout 4000 $PY -m tt_bio.main predict "$FASTA" \
    --model esmfold2 --fast --msa_dir $D/msa --seed 0 --max_msa_seqs "$2" \
    --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 \
    --out_dir "$D/out/$1" --override > "$D/run_$1.log" 2>&1
  echo "$1 exit=$? cif=$(ls $D/out/$1/*/structures/*.cif 2>/dev/null | wc -l)"
}
arm ${TAG}_full 8192
arm ${TAG}_cap5120 5120
$PY $WT/perf/esmfold2wh/domain_placement_ab.py "$EXP" \
  full=$(ls $D/out/${TAG}_full/*/structures/*.cif) cap5120=$(ls $D/out/${TAG}_cap5120/*/structures/*.cif) \
  $WT/perf/esmfold2wh/${TAG}_interdomain_ab.json
