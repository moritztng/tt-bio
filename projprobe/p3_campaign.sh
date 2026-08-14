#!/bin/sh
# Section 8.1 of state/relion-precision-fsc.md: the seed campaign the verdict still owes.
#
# ONE queue, ONE arm at a time, and nothing else may run on pc while it does. A box-256 arm
# holds 8.7 GB RSS (MEASURED) against ~9 GB available, so anything concurrent -- including a
# cheap box-96 arm -- pushes it over and the kernel kills it after every orientation and
# before it writes anything. That is not a hypothetical: it killed this campaign once at
# 04:03Z on seed 37, and `set -e` then took the remaining 22 arms down with it. Hence no
# `set -e`: an OOM must cost one arm, not the queue.
#
# Resumable: an arm whose JSON already exists is skipped, so a relaunch continues.
# Ordered so each seed lands a COMPLETE triple, then the paired tex8 comparison section 8.2
# calls "the one to quote", then the ladder, then the accumulator lower bound last -- the
# accumulator is the cheapest to justify dropping, because FLUSH=1 at box 96 already showed
# 25x more roundings does not cost more resolution.
cd "$(dirname "$0")" || exit 1
LOG=p3_campaign.log

run() {   # variant precision seed [flush]
  f=${4:-25}; tag=""; [ "$f" = 25 ] || tag="_f$f"
  out="p3fsc_box256_snr0.05_s$3_$1_$2$tag.json"
  if [ -f "$out" ]; then return 0; fi
  echo "=== $1/$2 seed $3 flush $f  $(date -u +%H:%M:%SZ)  avail $(awk '/MemAvailable/{print int($2/1048576)"G"}' /proc/meminfo)" >>$LOG
  if ! python3 p3_precision_fsc.py 256 400 0.05 "$3" "$1" "$2" "$f" >>$LOG 2>&1; then
    echo "!!! ARM FAILED: $out" >>$LOG
  fi
}

for s in 11 23 37 51 67 83 101 113; do
  run tri     bf16_dev      $s
  run twopass bf16_dev      $s
  run tri     bf16_dev_pess $s
done

# Section 8.2's secondary acceptance is "bf16_dev no worse than tex8, in the same harness",
# which needs tex8 on the same seeds or it is a paired n=8 against an unpaired n=1.
for s in 11 23 37 51 67 83 101 113; do
  run tri tex8 $s
done

# The ladder restated at the gate box rather than inherited from box 96 (pert1e0 is already
# measured at seed 11, +0.0250 A; these bracket it).
run tri pert1e-1 11
run tri pert3e0  11

# The accumulator lower bound: FLUSH=1 rounds the accumulator 400 times instead of 16.
run tri bf16acc 11 1
run tri bf16acc 23 1

echo "=== CAMPAIGN COMPLETE $(date -u +%H:%M:%SZ)" >>$LOG
