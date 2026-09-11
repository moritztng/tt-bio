#!/usr/bin/env bash
# b2x-integrate: re-verify both levers at 512 aa on one card, keeping every CIF.
set -eu
WT=/home/ttuser/.coworker/wt/b2x-integrate
cd "$WT"
exec env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2x-integrate \
  PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2x-flag-levers/ab_flag_levers.py \
  --out perf/b2x-integrate/ab512.json \
  --cifdir perf/b2x-integrate/cif \
  --reps 3 --keep-512 --skip-defect

# the two fixtures land in one --cifdir; split them so each scorer sees one fixture
mkdir -p perf/b2x-integrate/cif512 perf/b2x-integrate/cif298
mv perf/b2x-integrate/cif/512_* perf/b2x-integrate/cif512/ 2>/dev/null || true
mv perf/b2x-integrate/cif/298_* perf/b2x-integrate/cif298/ 2>/dev/null || true
