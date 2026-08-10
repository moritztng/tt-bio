#!/bin/sh
# qb1 leg of the reblock_permute landing gate: the release_gate legs qb2 could not reach because its
# w6_gate_msa cache holds one a3m against qb1's seven. Card 0, sequential, one device context at a
# time, a .done marker per step so a pass that ends mid-campaign resumes rather than repeats.
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-permute-flip-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1

export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-permute-flip-land
export PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export RELEASE_GATE_MSA_DIR=/home/ttuser/w6_gate_msa

L=perf/z_flip_land/logs_qb1
mkdir -p "$L"

# Assert the flip is live at the module constant, not under an env var, before anything is measured.
$PY -c 'import sys, tt_bio, tt_bio.reblock_permute as R
assert tt_bio.__file__.startswith("'"$WT"'"), tt_bio.__file__
assert R.REBLOCK_PERMUTE is True and R._ENABLED is True, (R.REBLOCK_PERMUTE, R._ENABLED)
import os; assert "TT_BIO_REBLOCK_PERMUTE" not in os.environ
print("preflight ok:", tt_bio.__file__)' > "$L/preflight.txt" 2>&1 || { echo PREFLIGHT_FAIL; cat "$L/preflight.txt"; exit 1; }
cat "$L/preflight.txt"

step() {
  name=$1; shift
  [ -f "$L/$name.done" ] && { echo "SKIP $name"; return; }
  echo "=== START $name $(date -u +%FT%TZ)"
  "$@" > "$L/$name.txt" 2>&1
  rc=$?
  echo "=== END $name rc=$rc $(date -u +%FT%TZ)"
  [ $rc -eq 0 ] && touch "$L/$name.done"
}

step gate_crossmodel $PY -u scripts/release_gate.py --model openfold3 --model boltz2 --model opendde
step gate_abag       $PY -u scripts/release_gate.py --model opendde-abag
step gate_capacity   $PY -u scripts/release_gate.py --model capacity
echo "QB1_GATE_ALL_DONE $(date -u +%FT%TZ)"
