#!/usr/bin/env bash
# Two-arm control of THIS branch's OPM flag at one model/rung, on one card, both arms in one script.
#
# Answers the question the size-ladder arm cannot answer right now: the ladder's verdict is
# dominated by a decline-clause baseline main never re-recorded, so a red there says nothing about
# this lever. This compares the lever census and the fold runtime with TT_BIO_OPM_LEGACY_LAYOUT
# off (the shipped fast path) against on (main's behaviour) at the SAME rung, so any difference is
# the flag by construction.
#
# Usage: MODEL=boltz2 RUNG=1024 CARD=1 bash opm_rung_control.sh
set -u
WT=/home/ttuser/.coworker/wt/c14-stack-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
MODEL=${MODEL:-boltz2}
RUNG=${RUNG:-1024}
CARD=${CARD:-1}
OUT=$WT/perf/c14_opm_layout/clause_control
mkdir -p "$OUT"
cd "$WT" || exit 2
export PYTHONPATH=$WT
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:c14-stack-land
FIX=perf/size512/fixtures/cdk2x2_${RUNG}.yaml
[ -f "$FIX" ] || { echo "no fixture $FIX"; exit 2; }
for legacy in 0 1; do
  label=${MODEL}-${RUNG}-opmlegacy${legacy}
  echo "=== $label start $(date -u +%FT%TZ) ==="
  rm -rf "$OUT/out_$label"
  TT_BIO_OPM_LEGACY_LAYOUT=$legacy "$PY" scripts/lever_census.py \
      --tt-bio "$PY" --label "$label" --out "$OUT/census_$label.json" -- \
      -m tt_bio.main predict "$FIX" \
      --model "$MODEL" --single_sequence --sampling_steps 6 --diffusion_samples 1 \
      --seed 0 --out_dir "$OUT/out_$label" > "$OUT/$label.log" 2>&1
  echo "=== $label rc=$? end $(date -u +%FT%TZ) ==="
done
