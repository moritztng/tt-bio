#!/bin/bash
# D56 fold check: TT_BIO_SOFTMAX_BW_RENORM on vs off on an OpenFold3 512 aa fold.
# The flag is read only inside backward closures, so all three legs must be byte-identical
# and TT_BIO_RENORM_STATS_DIR must stay empty of any non-zero count. Leg order on/off/on
# gives the A/A digest floor on the repeated arm.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/land_standing/out/d56
mkdir -p "$OUT"
cd "$WT" || exit 1

for leg in on off on2; do
  case $leg in
    on|on2) FLAG=1 ;;
    off)    FLAG=0 ;;
  esac
  rm -rf "$OUT/fold_$leg"
  mkdir -p "$OUT/stats_$leg"
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:land-standing \
  PYTHONPATH=$WT TT_BIO_SOFTMAX_BW_RENORM=$FLAG TT_BIO_RENORM_STATS_DIR="$OUT/stats_$leg" \
  "$PY" -m tt_bio.main predict "$WT/perf/size512/fixtures/cdk2x2_512.yaml" \
    --model openfold3 --single_sequence --sampling_steps 6 \
    --diffusion_samples 1 --seed 0 --out_dir "$OUT/fold_$leg" \
    > "$OUT/$leg.log" 2>&1
  echo "leg $leg rc=$? flag=$FLAG" >> "$OUT/RESULT.txt"
  find "$OUT/fold_$leg" -name '*.cif' -print0 | sort -z | xargs -0 -r sha256sum >> "$OUT/RESULT.txt"
  echo "stats: $(cat "$OUT/stats_$leg"/*.json 2>/dev/null | tr '\n' ' ')" >> "$OUT/RESULT.txt"
done
echo "ALL LEGS DONE $(date -u +%FT%TZ)" >> "$OUT/RESULT.txt"
