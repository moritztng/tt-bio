#!/usr/bin/env bash
# AF2-IG token bucketing (TT_BIO_AF2_BUCKET): device A/B, off / on / off2 per seed.
#
# AF2-IG folds PD-L1 + binder 80 at 195 tokens. With the flag on it pads to 224 (pad + mask +
# slice back), so the ragged-tile paths (SDPA pad, ragged reblock, T1/T2 declines at S % 32)
# give way to the whole-tile ones at 1.32x the pair work. Prints per-seed warm medians and the
# CA RMSD of each arm to off. Ship test: on not slower than off beyond the off vs off2 spread,
# and CA RMSD well inside the seed floor (0.6 A bar).
#
# Usage: [SEEDS="0 1 2"] perf/spd/af2_bucket_ab.sh <wh|bh> <card> [outdir]
set -euo pipefail
ARCH=${1:?wh|bh}; CARD=${2:?card}; OUT=${3:-out/af2_bucket_ab_$ARCH}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}
PARAMS=${PARAMS:-$HOME/.boltz/af2/params/params_model_1_ptm.npz}
FIX=${FIX:-$REPO/perf/kitcmp/fixtures}
mkdir -p "$OUT"
cd "$REPO"

for s in ${SEEDS:-0 1 2}; do
  for arm in off on off2; do
    case $arm in on) B=1 ;; *) B=0 ;; esac
    TT_VISIBLE_DEVICES=$CARD PYTHONPATH=$REPO TT_BIO_AF2_BUCKET=$B $PY perf/spd/af2ig_census.py \
      --input "$FIX/af2ig_pdl1_l80_s$s.yaml" --params "$PARAMS" --warm 2 --reps 4 --save-coords \
      --out "$OUT/s$s-$arm" > "$OUT/s$s-$arm.log" 2>&1
  done
done

$PY - "$OUT" ${SEEDS:-0 1 2} <<'EOF'
import json, statistics, sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1])
def ca(p):
    c = np.load(p).astype(np.float64)
    return c[..., 1, :].reshape(-1, 3) if c.ndim >= 3 and c.shape[-2] == 37 else c.reshape(-1, 3)
def rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))
for s in sys.argv[2:]:
    t = {a: statistics.median(json.loads((out / f"s{s}-{a}" / "census.json").read_text())["plain_fold_s"])
         for a in ("off", "on", "off2")}
    off = ca(out / f"s{s}-off" / "coords-0.npy")
    r = {a: rmsd(ca(out / f"s{s}-{a}" / "coords-0.npy"), off) for a in ("on", "off2")}
    print(f"s{s}: plain_fold_s off {t['off']:.3f} on {t['on']:.3f} off2 {t['off2']:.3f} | "
          f"CA RMSD to off: on {r['on']:.4f} off2 {r['off2']:.4f} A")
EOF
