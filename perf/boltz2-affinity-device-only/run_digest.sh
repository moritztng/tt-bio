#!/usr/bin/env bash
# Structure-model digest: determinism floor on the BEFORE tree, then the AFTER tree.
set -u
W=/home/ttuser/.coworker/wt/boltz2-affinity-device-only
O=$W/perf/boltz2-affinity-device-only
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FLAGS="--model boltz2 --single_sequence --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --seed 0"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-affinity-device-only
run() { # <tree> <label>
  local T=$1 L=$2 D
  D=$(mktemp -d /tmp/dig_${L}_XXXX)
  ( cd "$T" && PYTHONPATH="$T" $PY -m tt_bio.main predict "$T/examples/prot.yaml" --out_dir "$D" $FLAGS > "$D/run.log" 2>&1 )
  echo "$L rc=$? $(find $D -name "*.cif" | sort | xargs sha256sum 2>/dev/null | awk "{print \$1}" | tr "\n" " ") conf=$(find $D -name "confidence*.json" | sort | xargs sha256sum 2>/dev/null | awk "{print \$1}" | tr "\n" " ")" | tee -a $O/digest.txt
  rm -rf "$D"
}
run $W/.before-tree before_run1
run $W/.before-tree before_run2
run $W                after_run1
echo "DIGEST_EXIT=0 $(date -u +%FT%TZ)" >> $O/digest.txt
