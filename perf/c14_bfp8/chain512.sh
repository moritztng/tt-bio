#!/usr/bin/env bash
# The 512 aa half, all clock-insensitive: the kernel-path census at the CELL size (the 298 aa one
# cannot stand in for it -- b2z's `transition` site scored 0.266 A at 298 and moved the 512 aa fold
# 13.21 A) plus the structures the Angstrom reading needs.
set -u
cd /home/ttuser/.coworker/wt/c14-bfp8-fastpath || exit 1
P=/home/ttuser/tt-bio-dev/env/bin/python3
E="env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:c14-bfp8-fastpath"
NOISE='DEBUG|Config\{|UMD \||leaked|nanobind|^ - |^See http|^ --- '
run() { local label="$1" n="$2"; shift 3
  echo "=== STEP $label $(date -uIs)"; "$@" 2>&1 | grep -avE "$NOISE" | tail -"$n"
  echo "=== STEP $label rc=${PIPESTATUS[0]} $(date -uIs)"; }

for arm in off on aa; do
  case $arm in
    on) FL=(env TT_BIO_TRIATT_B8=1) ;;
    *)  FL=(env -u TT_BIO_TRIATT_B8) ;;     # `aa` is a second base fold: the A/A control
  esac
  run 5_census512_$arm 14 -- $E "${FL[@]}" $P scripts/lever_census.py --tt-bio $P \
    --label triatt_b8_512_$arm --out perf/c14_bfp8/census_512_$arm.json \
    -- -m tt_bio.main predict perf/size512/fixtures/cdk2x2_512.yaml \
    --sampling_steps 200 --recycling_steps 3 --seed 0 \
    --out_dir perf/c14_bfp8/census512_out_$arm
done
echo "=== CHAIN512 DONE $(date -uIs)"
