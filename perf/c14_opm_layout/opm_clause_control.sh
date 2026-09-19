#!/usr/bin/env bash
# Is the size-ladder boltz2/256 decline-clause drift this branch's OPM flag, or main's trimul?
# One rung, both arms of the flag, on the free sibling card. A census is a correctness read, not a
# timing read, so a busy sibling cannot change the answer.
set -u
WT=/home/ttuser/.coworker/wt/c14-stack-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c14_opm_layout/clause_control
mkdir -p "$OUT"
cd "$WT" || exit 2
export PYTHONPATH=$WT
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:c14-stack-land
for legacy in 0 1; do
  label=boltz2-256-opmlegacy$legacy
  echo "=== $label start $(date -u +%FT%TZ) ==="
  rm -rf "$OUT/out_$label"
  TT_BIO_OPM_LEGACY_LAYOUT=$legacy "$PY" scripts/lever_census.py \
      --tt-bio "$PY" --label "$label" --out "$OUT/census_$label.json" -- \
      -m tt_bio.main predict perf/size512/fixtures/cdk2x2_256.yaml \
      --model boltz2 --single_sequence --sampling_steps 6 --diffusion_samples 1 \
      --seed 0 --out_dir "$OUT/out_$label" > "$OUT/$label.log" 2>&1
  echo "=== $label rc=$? end $(date -u +%FT%TZ) ==="
done
