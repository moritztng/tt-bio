#!/bin/bash
# SaProt --structure cases with a foldseek binary, which whglx does not have on PATH:
# run_np_extra.sh <card> <foldseek>. Same layout as run_np.sh, under out_np/saprot/<model>/.
set -u
cd "$(dirname "$0")/../.."
C=$1 FS=$2
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mgx-matrix
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxm
export TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_TIMEOUT=${TT_BIO_LEASE_TIMEOUT:-1800}
O=perf/mgx_matrix/out_np
ref=$(ls perf/mgx_matrix/out/protenix-v2/*_results_*/structures/base.cif | head -1)
for m in saprot-35m saprot-650m saprot-1.3b; do
  for c in "structure_match $ref" "structure_mismatch examples/ground_truth_structures/prot.cif"; do
    set -- $c; tag=saprot/$m/$1; start=$(date +%s)
    $HOME/env/bin/python -m tt_bio.main saprot "$O/ubq.fasta" --model $m --structure "$2" \
        --foldseek "$FS" --out_dir "$O/$tag" > "$O/$tag.log" 2>&1
    echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$O/$tag.log"
  done
done
