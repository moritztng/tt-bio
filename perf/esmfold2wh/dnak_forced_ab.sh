#!/bin/bash
# The 8192-vs-5120 A/B at the brief own length, on the brief own cleanest fixture:
# E. coli DnaK 1-605 (PDB 2KHO), whose two domains are the only candidate that splits into
# exactly one segment each (4-384 / 385-603) with their centroids 63.5 A apart.
# Wormhole is not available -- GWH02 UMD 28 is held by a live JapanFold prod worker -- so
# both arms run on qb1 card 2 with the small-grid budgets forced (force_small_grid.py).
set -u
WT=/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-msa-cap-p3-prove
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=/home/ttuser/msacap_p3
cd "$WT" || exit 1
arm() {
  TT_VISIBLE_DEVICES=2 TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:japanfold-esmfold2-wh-msa-cap-p3-prove \
  PYTHONPATH=$WT timeout 4000 $PY $WT/perf/esmfold2wh/force_small_grid.py predict $D/dnak_2kho.fasta \
    --model esmfold2 --fast --msa_dir $D/msa --seed 0 --max_msa_seqs "$2" \
    --recycling_steps 10 --sampling_steps 100 --diffusion_samples 1 \
    --out_dir "$D/out/$1" --override > "$D/run_$1.log" 2>&1
  echo "$1 exit=$? cif=$(ls $D/out/$1/*/structures/*.cif 2>/dev/null | wc -l)"
}
arm dnak_full 8192
arm dnak_cap5120 5120
$PY $WT/perf/esmfold2wh/domain_placement_ab.py $D/pdb/2KHO.cif \
  full=$(ls $D/out/dnak_full/*/structures/*.cif) cap5120=$(ls $D/out/dnak_cap5120/*/structures/*.cif) \
  $WT/perf/esmfold2wh/dnak_interdomain_ab.json
