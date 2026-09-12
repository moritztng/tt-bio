#!/usr/bin/env python3
"""Paired interleaved A/B of the atom-attention key gather, Boltz-2 512 aa shape.

The shipped gather (tt_bio/tenstorrent.py, AttentionPairBias atom_level branch) turns a
stride-32 sliding window into a one-hot matmul, and pays four arithmetic-free programs to do
it: a reshape that splits every 32-atom window into two 16-row halves (doubling the tiled
footprint, because 16 pads to 32), a permute that drags the half-window axis to the end so the
matmul can contract over it, the inverse permute, and a reshape that glues the 8 gathered
half-blocks back into a 128-row key window. On Blackhole those four cost 2.3045 ms of a
22.0224 ms diffusion step -- 45 % of every arithmetic-free program in the step.

Arm B deletes all five. The keys of window k are s_flat[32k-48 : 32k+80], so shifting the flat
atom axis by 48 rows makes every one of the four 32-row pieces land on a window boundary, and
the gather becomes four window-axis slices and one concat. Pure index motion: bit-exact.

Arm C is the same, doing the one unaligned shift through ROW_MAJOR instead of a tiled pad, in
case the tiled pad is the expensive part.

n>=7, arms interleaved rep by rep in ONE process, A/A floor reported.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

W = 32          # ATOM_WINDOW
KEYS = 128      # ATOM_DIM
SHIFT = 48      # keys of window k start at 32k - 48


def build_keys_indexing(K):
    """The shipped one-hot: half-block (2k-3+j) of the source feeds half-block (8k+j) of dest."""
    import torch
    ki = torch.zeros(2 * K, 8 * K)
    for k in range(K):
        for j in range(8):
            src = 2 * k - 3 + j
            if 0 <= src < 2 * K:
                ki[src, 8 * k + j] = 1.0
    return ki


def arm_a(ttnn, s, ki, ckc, grid):
    B, K, w, D = s.shape
    x = ttnn.reshape(s, (B, 2 * K, w // 2, -1))
    x = ttnn.permute(x, (0, 2, 3, 1))
    x = ttnn.matmul(x, ki, compute_kernel_config=ckc, core_grid=grid)
    x = ttnn.permute(x, (0, 3, 1, 2))
    return ttnn.reshape(x, (B, K, -1, D))


def arm_b(ttnn, s, ki, ckc, grid):
    """Tiled front-pad by 48, four window-axis slices, one concat."""
    B, K, w, D = s.shape
    flat = ttnn.reshape(s, (B, 1, K * w, D))
    t = ttnn.pad(flat, [(0, 0), (0, 0), (SHIFT, 4 * w - SHIFT), (0, 0)], 0.0)
    tw = ttnn.reshape(t, (B, K + 4, w, D))
    return ttnn.concat([tw[:, j:j + K] for j in range(4)], dim=2)


def arm_c(ttnn, s, ki, ckc, grid):
    """Same, with the one unaligned shift taken through ROW_MAJOR."""
    B, K, w, D = s.shape
    flat = ttnn.reshape(s, (B, 1, K * w, D))
    rm = ttnn.to_layout(flat, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.pad(rm, [(0, 0), (0, 0), (SHIFT, 4 * w - SHIFT), (0, 0)], 0.0)
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, dtype=s.dtype)
    tw = ttnn.reshape(t, (B, K + 4, w, D))
    return ttnn.concat([tw[:, j:j + K] for j in range(4)], dim=2)


ARMS = {"A": arm_a, "B": arm_b, "C": arm_c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, default=4480)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T

    K = a.atoms // W
    out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "atoms": a.atoms, "windows": K, "dim": a.dim, "reps": a.reps,
           "inner": a.inner, "t0": time.time()}

    T.get_device(trace_region_size=1 << 29)
    dev = T.get_device()
    grid = T.CORE_GRID_MAIN
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)
    s_pt = torch.randn(1, K, W, a.dim)
    s = ttnn.from_torch(s_pt, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ki = ttnn.from_torch(build_keys_indexing(K), device=dev, layout=ttnn.TILE_LAYOUT,
                         dtype=ttnn.bfloat4_b)

    arms = [x for x in a.arms.split(",") if x]

    # Parity first: every arm must reproduce arm A exactly.
    ref = ttnn.to_torch(ARMS["A"](ttnn, s, ki, ckc, grid))
    out["ref_shape"] = list(ref.shape)
    out["parity"] = {}
    for name in arms:
        if name == "A":
            continue
        try:
            got = ttnn.to_torch(ARMS[name](ttnn, s, ki, ckc, grid))
            out["parity"][name] = {
                "shape": list(got.shape),
                "equal": bool(got.shape == ref.shape and torch.equal(got, ref)),
                "max_abs": float((got - ref).abs().max()) if got.shape == ref.shape else None,
            }
        except Exception as e:                                    # noqa: BLE001
            out["parity"][name] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out["parity"], indent=1), flush=True)
    arms = [n for n in arms if n == "A" or out["parity"].get(n, {}).get("equal")]
    print(f"arms surviving parity: {arms}", flush=True)

    def timed(name):
        fn = ARMS[name]
        for _ in range(3):
            ttnn.deallocate(fn(ttnn, s, ki, ckc, grid))
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(a.inner):
            ttnn.deallocate(fn(ttnn, s, ki, ckc, grid))
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t) * 1e3 / a.inner

    series = {n: [] for n in arms}
    for r in range(a.reps):
        for n in arms:
            series[n].append(timed(n))
        print(f"rep {r}: " + "  ".join(f"{n}={series[n][-1]:.4f}" for n in arms), flush=True)

    med = {n: statistics.median(v) for n, v in series.items()}
    out["series"] = series
    out["median_ms"] = med
    out["ratio_vs_A"] = {n: med["A"] / med[n] for n in arms}
    # A/A floor: two more independent A measurements, same interleaving.
    floor = [timed("A") for _ in range(2)]
    out["aa_floor"] = [med["A"] / f for f in floor]
    out["loadavg"] = [round(x, 2) for x in os.getloadavg()]
    print(json.dumps({k: out[k] for k in ("median_ms", "ratio_vs_A", "aa_floor", "loadavg")},
                     indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
