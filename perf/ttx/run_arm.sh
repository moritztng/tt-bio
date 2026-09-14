#!/usr/bin/env bash
# One stack arm of the Blackhole shared-draw fold, run out-of-tree so the pin never moves.
#
#   run_arm.sh <new|old|VERSION> <card> <sizes> <outdir>
#
# new = /home/ttuser/scratch/ttx/venv-new (ttnn 0.78.0 + SFPI 7.72.0 in its own runtime/)
# old = /home/ttuser/tt-bio-dev/env      (ttnn 0.68.0, the pin, used read-only)
# VERSION = /home/ttuser/scratch/ttx/venv-<VERSION>, built by mkvenv.sh, for the version bisect
#
# Both arms run with every hand-written fused kernel OFF: they do not compile against 0.78's LLK
# surface, and the old arm with them off folds byte-identically to the old arm with them on
# (b2z-ttnn-upgrade section 8.3), so the configuration is numerically the shipped one.
set -euo pipefail
ARM=$1; CARD=$2; SIZES=$3; OUT=$4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
case "$ARM" in
  new) PY=/home/ttuser/scratch/ttx/venv-new/bin/python ;;
  old) PY=/home/ttuser/tt-bio-dev/env/bin/python ;;
  *)   PY=/home/ttuser/scratch/ttx/venv-$ARM/bin/python ;;   # a bisect arm, mkvenv.sh <version>
esac
[ -x "$PY" ] || { echo "no interpreter for arm $ARM at $PY" >&2; exit 2; }
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:ttx-version-recon
export PYTHONPATH=$REPO
export TT_METAL_CACHE=/home/ttuser/scratch/ttx/cache-$ARM
export TT_MESH_GRAPH_DESC_PATH=$REPO/perf/ttx/mgd/bh_1x1.textproto
export TT_BIO_TRIATT_PERSISTENT_MASK=0 TT_BIO_TRIATT_HEAD_MAJOR_QKV=0 \
       TT_BIO_TRIATT_HEAD_MAJOR_TAIL=0 TT_BIO_REBLOCK_PERMUTE_BACK=0 \
       TT_BIO_REBLOCK_PERMUTE_GATED=0 TT_BIO_TRIMUL_DUAL_NOC=0
exec "$PY" "${TTX_FOLD_ENTRY:-$REPO/perf/roof_shared/fold_shared.py}" \
  --out "$OUT/folds.json" --cifdir "$OUT/cif" --sizes "$SIZES" --arms shared --seed 0
