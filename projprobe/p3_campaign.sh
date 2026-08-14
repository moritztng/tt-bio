#!/bin/sh
# Section 8.1 of state/relion-precision-fsc.md: the seed campaign the verdict still owes.
# Runs one arm at a time -- a box-256 arm holds ~8.5 GB and pc has ~16 GB available, so two
# concurrent arms swap or get OOM-killed after doing all the work and before writing anything.
# Ordered so that each seed lands a COMPLETE triple: a turn that runs out of time still leaves
# a paired n, not three ragged half-campaigns.
set -e
cd "$(dirname "$0")"
LOG=p3_campaign.log

run() {   # variant precision seed [flush]
  f=${4:-25}; tag=""; [ "$f" = 25 ] || tag="_f$f"
  out="p3fsc_box256_snr0.05_s$3_$1_$2$tag.json"
  if [ -f "$out" ]; then echo "skip $out (exists)" >>$LOG; return 0; fi
  echo "=== $1/$2 seed $3 flush $f  $(date -u +%H:%M:%SZ)" >>$LOG
  python3 p3_precision_fsc.py 256 400 0.05 "$3" "$1" "$2" "$f" >>$LOG 2>&1
}

for s in 23 37 51 67 83 101 113; do
  run tri     bf16_dev      $s
  run twopass bf16_dev      $s
  run tri     bf16_dev_pess $s
done

# The ladder restated at the gate box rather than inherited from box 96 (pert1e0 is already
# measured at seed 11, +0.0250 A; these bracket it).
run tri pert1e-1 11
run tri pert3e0  11

# The accumulator lower bound: FLUSH=1 rounds the accumulator 400 times instead of 16.
run tri bf16acc 11 1
run tri bf16acc 23 1

echo "=== CAMPAIGN COMPLETE $(date -u +%H:%M:%SZ)" >>$LOG
