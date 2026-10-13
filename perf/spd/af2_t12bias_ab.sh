#!/usr/bin/env bash
# AF2 trimul T1/T2 with biases: device A/B. WRITTEN, NOT RUN (no device was used to write it).
#
# What it grades: commit "Let trimul T1 and T2 carry projection biases for AF2" (wk/spd-gap-bind).
# T1 (fused in-projection) now carries AF2's p_in / g_in / g_out biases and T2 (fused tail) its
# p_out bias, each seeded into the fp32 DST accumulator before the K loop (trimul_gin_ragged's
# technique). AF2's default lever set is NOT changed; this job decides whether it should be.
#
# KNOWN BEFORE RUNNING: AF2-IG's pdl1 fixtures fold at 195 tokens, and T1 reads whole tiles only
# (trimul_tail.t1_eligible -> "act_shape" when S % 32 != 0; pinned in
# tests/test_trimul_bias_t1t2.py::test_t1_still_declines_ragged_af2_size). T2 only runs on T1's
# g_out, so at 195 the t12b arm is expected to serve NOTHING and to read bit-identical to own
# (coords_sha16 equal). Step 0 below confirms that through the decline counters; a real speed
# A/B needs either a whole-tile AF2 input (224 tokens) or a ragged T1 (not written).
#
# Hosts / cards (pick free ones through the fleet, never a cardblocked card):
#   WH : a .114 pool chip     ARMS own=triatt_tail_bias,pair_mm
#   BH : a qb1 p150a card     ARMS own=triatt_tail_bias
# Usage: [SIZES="224"] [SEEDS=""] perf/spd/af2_t12bias_ab.sh <wh|bh> <card> [outdir]
set -euo pipefail
ARCH=${1:?wh|bh}; CARD=${2:?card}; OUT=${3:-out/af2_t12bias_ab_$ARCH}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}
PARAMS=${PARAMS:-$HOME/.boltz/af2/params/params_model_1_ptm.npz}
# The fixtures are not in git (af2ig_census.py documents this path); point FIX at the host copy.
FIX=${FIX:-$REPO/perf/kitcmp/fixtures}
case $ARCH in
  wh) OWN=triatt_tail_bias,pair_mm ;;
  bh) OWN=triatt_tail_bias ;;
  *) echo "arch wh|bh" >&2; exit 2 ;;
esac
T12B=$OWN,trimul_t1,trimul_t2,trimul_t1_dram
mkdir -p "$OUT"
cd "$REPO"

# 1. microbench: T1 / T2 outputs vs a float64 torch reference of the biased unfused math, against
#    the unfused ttnn path's own error vs the same reference. Pass: fused rel L2 within noise of
#    (<= 1.1x) the unfused rel L2 for a, b, g_out and the T2 residual update. At 195 it records the
#    decline; at 224 (whole tiles) it grades the kernels.
for S in ${SIZES:-195 224}; do
  TT_VISIBLE_DEVICES=$CARD PYTHONPATH=$REPO TT_BIO_LEVERS=$T12B $PY - "$S" <<'EOF' | tee "$OUT/micro_$S.txt"
import sys, torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TTL, mm_generic as _mm_generic
S, K, H, N = int(sys.argv[1]), 128, 128, 128
torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
r = lambda *s, sc=1.0: (torch.randn(*s) * sc).to(torch.bfloat16).double()
x, lw, lb = r(1, S, S, K), 1 + r(K, sc=0.1), r(K, sc=0.1)
wp, wg, wo = r(K, 2 * H, sc=K ** -0.5), r(K, 2 * H, sc=K ** -0.5), r(K, N, sc=K ** -0.5)
bp, bg, bgo = r(2 * H, sc=0.5), r(2 * H, sc=0.5), r(N, sc=0.5)
wpo, bpo = r(H, N, sc=H ** -0.5), r(N, sc=0.5)
xn = torch.nn.functional.layer_norm(x, (K,), lw, lb, 1e-5)
ab = (xn @ wp + bp) * torch.sigmoid(xn @ wg + bg)
ref_a, ref_b = ab[..., :H].permute(0, 3, 1, 2), ab[..., H:].permute(0, 3, 1, 2)
ref_go = xn @ wo + bgo
rel = lambda got, ref: float((got.double() - ref).norm() / ref.norm())
ckc = _mm_generic.ckc_args(ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True))
xt = up(x)
lwt, lbt = up(lw.reshape(1, K)), up(lb.reshape(1, K))
grid = tuple(T.COMPUTE_GRID_MAIN)
bias = TTL.t1_bias_tiles(bp, bg, bgo, S, dev)
with T.levers("trimul_t1,trimul_t2"):
    outs = TTL.fused_in_ln(xt, lwt, lbt, up(wp.t().contiguous()), up(wg.t().contiguous()), up(wo),
                           ckc, grid, bias=bias)
print("T1 served" if outs else f"T1 declined: {TTL.REJECTS}")
# the unfused reference path: what TriangleMultiplication runs without T1
xnt = ttnn.layer_norm(xt, weight=lwt, bias=lbt, epsilon=1e-5)
lin = lambda w, b: ttnn.linear(xnt, up(w), bias=up(b.reshape(1, -1)))
ua = ttnn.multiply(lin(wp, bp), ttnn.sigmoid(lin(wg, bg)))
ua = ttnn.to_torch(ua).double()
u_a, u_b = ua[..., :H].permute(0, 3, 1, 2), ua[..., H:].permute(0, 3, 1, 2)
u_go = ttnn.to_torch(lin(wo, bgo)).double()
print(f"unfused rel L2  a {rel(u_a, ref_a):.3e} b {rel(u_b, ref_b):.3e} g_out {rel(u_go, ref_go):.3e}")
if outs:
    fa, fb, fgo = (ttnn.to_torch(o).double() for o in outs)
    print(f"T1      rel L2  a {rel(fa, ref_a):.3e} b {rel(fb, ref_b):.3e} g_out {rel(fgo, ref_go):.3e}")
    # T2 on a pair-shaped tail input, the T1 g_out as its gate, p_out's bias seeded
    xo, z = r(1, S, S, H), r(1, S, S, N)
    xon = torch.nn.functional.layer_norm(xo, (H,), lw, lb, 1e-5)
    ref_z = z + (xon @ wpo + bpo) * torch.sigmoid(fgo)
    zt = up(z)
    got = TTL.fused_tail_ln(up(xo), lwt, lbt, up(wpo), outs[2], zt, ckc, grid,
                            pbias=TTL.t2_bias_tiles(bpo, dev))
    if got is None:
        print(f"T2 declined: {TTL.REJECTS}")
    else:
        xont = ttnn.layer_norm(up(xo), weight=lwt, bias=lbt, epsilon=1e-5)
        uz = ttnn.to_torch(ttnn.add(up(z), ttnn.multiply(
            ttnn.linear(xont, up(wpo), bias=up(bpo.reshape(1, -1))), ttnn.sigmoid(outs[2])))).double()
        d = lambda t: rel(t - z, ref_z - z)          # error of the update, not of z
        print(f"T2 rel L2 (update) fused {d(ttnn.to_torch(got).double()):.3e} unfused {d(uz):.3e}")
ttnn.close_device(dev)
EOF
done

# SEEDS= (empty) stops after the microbench.
[ -n "${SEEDS-0 1 2}" ] || exit 0
# 2. AF2-IG census A/B, three seeds, arms own / t12b / own2 (own2 = own again: the noise floor).
for s in ${SEEDS-0 1 2}; do
  for arm in own t12b own2; do
    case $arm in t12b) L=$T12B ;; *) L=$OWN ;; esac
    TT_VISIBLE_DEVICES=$CARD PYTHONPATH=$REPO TT_BIO_LEVERS=$L $PY perf/spd/af2ig_census.py \
      --input "$FIX/af2ig_pdl1_l80_s$s.yaml" --params "$PARAMS" --warm 2 --reps 4 --save-coords \
      --out "$OUT/s$s-$arm" > "$OUT/s$s-$arm.log" 2>&1
  done
done

# 3. read: median plain_fold_s per arm, t12b-vs-own and own2-vs-own CA RMSD (Kabsch), sha equality.
$PY - "$OUT" <<'EOF'
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
for s in (0, 1, 2):
    row = {}
    for arm in ("own", "t12b", "own2"):
        j = json.loads((out / f"s{s}-{arm}" / "census.json").read_text())
        row[arm] = (statistics.median(j["plain_fold_s"]), j["coords_sha16"])
    own = ca(out / f"s{s}-own" / "coords-0.npy")
    print(f"s{s}: plain_fold_s median own {row['own'][0]:.3f} t12b {row['t12b'][0]:.3f} "
          f"own2 {row['own2'][0]:.3f} | CA RMSD t12b {rmsd(ca(out / f's{s}-t12b' / 'coords-0.npy'), own):.4f} "
          f"own2 {rmsd(ca(out / f's{s}-own2' / 'coords-0.npy'), own):.4f} A | "
          f"sha t12b==own {row['t12b'][1] == row['own'][1]}")
EOF
echo "Also grep each t12b log for the T1/T2 counters (T1_STATS / T1_WHY / T2_STATS) to see what served."
