#!/bin/sh
# Run one arm of the FIX-2 cross-model test. Working-tree-only reverse patches, never committed.
# Per-step `.done` markers so a pass that ends mid-arm leaves finished steps intact.
#
#   sh perf/z_fix2_crossmodel/run_arm.sh A
set -e
ARM="$1"
[ -n "$ARM" ] || { echo "usage: run_arm.sh <A|B|C|D>"; exit 2; }

WT=/home/ttuser/.coworker/wt/protenix-trunk--z-fix2-crossmodel
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D="$WT/perf/z_fix2_crossmodel"
cd "$WT"

export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-fix2-crossmodel
export PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export RELEASE_GATE_MSA_DIR=/home/ttuser/w6_gate_msa

git checkout -- tt_bio/tenstorrent.py
case "$ARM" in
  A) git apply -R "$D/fixd.patch" "$D/fix2.patch" ;;
  B) git apply -R "$D/fixd.patch" ;;
  C) : ;;
  D) git apply -R "$D/fix2.patch" ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
echo "=== arm $ARM working tree ==="
git diff --stat tt_bio/tenstorrent.py

T298=perf/size512/fixtures/cdk2x2_298.yaml
T512=perf/size512/fixtures/cdk2x2_512.yaml

step() {   # step <model> <targets>
  m="$1"; t="$2"
  mk="$D/.done_${ARM}_${m}"
  if [ -f "$mk" ]; then echo "skip $ARM/$m (done)"; return 0; fi
  echo "=== arm $ARM  model $m  targets $t ==="
  $PY -u "$D/arm_sweep.py" --arm "$ARM" --model "$m" --targets "$t" \
      --out "$D/arm${ARM}_${m}.json" 2>&1 | tee "$D/arm${ARM}_${m}.log"
  touch "$mk"
}

step openfold3 "$T298,$T512"
step boltz2    "$T298,$T512"
case "$ARM" in
  A|C) step opendde "$T512" ;;
esac

git checkout -- tt_bio/tenstorrent.py
echo "=== arm $ARM complete, tree restored ==="
