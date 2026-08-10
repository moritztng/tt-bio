#!/bin/bash
# boltz2-hsa-nomsa, scored the way its fixture actually supports.
#
# Under the global --legacy-rdx the gate forces every leg down the legacy R/D/X path, which wants a
# `settings_tag` in the fixture's meta.json. boltz2/hsa is an ENVELOPE-only fixture (ref_fp32 +
# ref_bf16, no settings_tag), so that path errors with "reference fixture settings-tag mismatch ...
# meta.json says None" and the leg comes back ERROR on any arm -- it is a leg/flag mismatch, not a
# result. The protenix and opendde legs in the same sweep are harvested fixtures and do carry the
# tag, which is why they score fine.
#
# So run this one leg WITHOUT --legacy-rdx, against its own envelope references. That turns a
# dropped leg into a real verdict instead of a hole in the coverage.
#   bash perf/w6_c2fix/boltz2env.sh <BASE|C2FIX>
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-c2fix-land || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-c2fix-land
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
ARM=$1
BASE_REF=$(sed -n 's/^BASE_REF = "\(.*\)"/\1/p' perf/w6_c2fix/arm.py)
[ -n "$BASE_REF" ] || exit 1
[ -s "perf/w6_c2fix/out/fpgenv_${ARM}.json" ] && { echo "SKIP boltz2env $ARM"; exit 0; }
$PY perf/w6_c2fix/arm.py --arm "$ARM" >/dev/null || exit 1
$PY scripts/full_parity_gate.py --workers tt-quietbox:1 --leg boltz2-hsa-nomsa \
    --workdir "$HOME/c2fix_env_${ARM}_${BASE_REF}" \
    --out perf/w6_c2fix/out/fpgenv_${ARM}.json 2>&1 | tee perf/w6_c2fix/out/fpgenv_${ARM}.log
echo "BOLTZ2ENV DONE $ARM $(date -u +%H:%M:%S)"
