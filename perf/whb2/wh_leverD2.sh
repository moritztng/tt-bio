#!/bin/bash
# Lever D, measured through the production CLI instead of the campaign harness.
#
# xmodel_ab cannot measure this. It passes --samples into job_cfg and tt_baseline carries it into
# predict_one, but the fold writes ONE structure and the wall does not move: B=1, 2 and 4 at 298 aa
# all returned 24.8-24.9 s with one CIF and pLDDT identical to six decimals. VERIFIED against the
# production path, which does honour it -- `tt_bio.main predict --diffusion_samples 1` writes 1 CIF
# and `4` writes 4. So the capability is real and the harness silently collapses it, which would
# have made any multiplicity number taken there a fabricated 1.0x.
#
# Production settings, so this answers what the service can actually offer: 3 recycles, 200 steps.
# Each arm counts its own CIF files; an arm whose count != B is not reported as a timing.
set -u
: "${TREE:?}" "${OUT:?}"
mkdir -p "$OUT"
cd "$TREE" || exit 1
. /home/cust-team/mthuening/whbase/pick_card.sh

for S in 298 512; do
  for B in 1 2 4 5; do
    C=$(pick_card) || { echo "no free card for $S/B$B"; continue; }
    d="$OUT/s${S}_b${B}"
    rm -rf "$d" && mkdir -p "$d"
    t0=$(date +%s.%N)
    TT_VISIBLE_DEVICES=$C TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 \
      ./env/bin/python -m tt_bio.main predict "perf/size512/fixtures/cdk2x2_${S}.yaml" \
        --model boltz2 --out_dir "$d" --override --seed 1 --single_sequence \
        --recycling_steps 3 --sampling_steps 200 --diffusion_samples "$B" \
        > "$OUT/s${S}_b${B}.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    n=$(find "$d" -name '*.cif' | wc -l)
    echo "RESULT size=$S B=$B rc=$rc wall=$(echo "$t1 - $t0" | bc) cifs=$n card=$C"
    [ "$rc" -ne 0 ] && { echo "size=$S stops at B=$B"; break; }
  done
done
echo "LEVER D2 DONE $(date -u +%H:%M:%S)"
