#!/usr/bin/env python3
"""P5 / C1 — the OuterProductMean layout chain, and whether the relayout is intrinsic.

Probe only; nothing under `tt_bio/` is touched.

Production (`tt_bio/tenstorrent.py`, this branch): `matmul:3037` emits `(9536, 9536)` from
`a_flat (I*C, S)` and `b_flat (D*J, S)`, then `to_layout:3059`, `reshape:3060`, `to_layout:3061`,
`permute:3062` relayout it to `(298, 320, 1024)` for `linear:3068`.

Index algebra: the matmul's M axis carries (i, c) and its N axis carries (d, j); the consumer needs
(i, j) contiguous and (c, d) contracted. So no matmul that keeps `a` on M and `b` on N can emit the
consumer's layout -- the question is which producer layout leaves the CHEAPEST residual relayout.

Arms:
  A0  baseline, every op timed on its own
  A1  b-side reorder: build b_flat as (j*D+d, S) instead of (d*J+j, S), so the matmul's output tile
      (i, j) IS the (c, d) block. Matmul shapes are identical -- the kill test is whether its time moves.
  A2  batched over i: a (I, C, S) @ (S, D*J) -> (I, C, D*J). What it costs the matmul to emit a
      different layout, directly.
  A3  the single-pass floor: one DRAM->DRAM clone of the same 182 MB, and a single-op transpose.
"""
from __future__ import annotations

import argparse, json, statistics as st, sys, time
from pathlib import Path

import torch, ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=2, pipe=3, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def us(s):
    return round(s * 1e6, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", type=int, default=298)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--s", type=int, default=35)
    ap.add_argument("--cz", type=int, default=256)
    ap.add_argument("--calls", type=int, default=40, help="OPM calls per fold (4 blocks x 10 cycles)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    I = J = args.i
    C = D = args.c
    S = args.s
    K = args.calls
    res = {"shape": {"I": I, "J": J, "C": C, "D": D, "S": S, "c_z": args.cz,
                     "calls_per_fold": K, "conversion": "us/call x 40 / 1000 = ms/fold"},
           "device": {"host": "qb2", "card": 1, "ttnn": "0.68.0",
                      "grid": f"{dev.compute_with_storage_grid_size().x}x"
                              f"{dev.compute_with_storage_grid_size().y}"}}

    torch.manual_seed(0)
    a_t = torch.randn(S, I, C) * 0.1          # project_a output, (S, I, C)
    b_t = torch.randn(S, J, D) * 0.1          # project_b output, (S, J, D)
    ow = ttnn.from_torch(torch.randn(C * D, args.cz) * 0.05, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)

    def upload(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=DRAM)

    # ---- operands, built exactly as production does ------------------------------------------
    a = ttnn.permute(upload(a_t), (1, 2, 0))                      # (I, C, S)
    a_flat = ttnn.reshape(a, (I * C, S))                          # (i*C+c, s)

    def b_flat_of(perm):
        b = ttnn.permute(upload(b_t), perm)
        b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT)
        b = ttnn.reshape(b, (-1, S))
        return ttnn.to_layout(b, ttnn.TILE_LAYOUT)

    b_dj = b_flat_of((2, 1, 0))     # (D, J, S) -> (d*J+j, s)   -- production
    b_jd = b_flat_of((1, 2, 0))     # (J, D, S) -> (j*D+d, s)   -- arm A1

    # ---- A0: the production chain, op by op ---------------------------------------------------
    print("=== A0 baseline chain ===", flush=True)
    a0 = {}
    a0["matmul_3037"] = us(timed(dev, lambda: ttnn.deallocate(
        ttnn.matmul(a_flat, b_dj, transpose_b=True, compute_kernel_config=ckc))))
    z = ttnn.matmul(a_flat, b_dj, transpose_b=True, compute_kernel_config=ckc)
    print(f"  matmul -> {tuple(z.shape)} {a0['matmul_3037']} us", flush=True)

    a0["to_layout_RM_3059"] = us(timed(dev, lambda: ttnn.deallocate(
        ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT))))
    zr = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    a0["reshape_3060"] = us(timed(dev, lambda: ttnn.deallocate(
        ttnn.reshape(zr, (I, C * D, J)))))
    zs = ttnn.reshape(zr, (I, C * D, J))
    a0["to_layout_TILE_3061"] = us(timed(dev, lambda: ttnn.deallocate(
        ttnn.to_layout(zs, ttnn.TILE_LAYOUT))))
    zt = ttnn.to_layout(zs, ttnn.TILE_LAYOUT)
    a0["permute_3062"] = us(timed(dev, lambda: ttnn.deallocate(
        ttnn.permute(zt, (0, 2, 1)))))
    zp = ttnn.permute(zt, (0, 2, 1))
    print(f"  relayout -> {tuple(zp.shape)}", flush=True)
    a0["linear_3068"] = us(timed(dev, lambda: ttnn.deallocate(
        ttnn.linear(zp, ow, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)), 2, 2, 5))
    a0["chain4_us"] = round(a0["to_layout_RM_3059"] + a0["reshape_3060"]
                            + a0["to_layout_TILE_3061"] + a0["permute_3062"], 1)
    a0["chain4_ms_per_fold"] = round(a0["chain4_us"] * K / 1000, 1)
    a0["matmul_ms_per_fold"] = round(a0["matmul_3037"] * K / 1000, 1)
    res["A0_baseline"] = a0
    print("  " + json.dumps(a0), flush=True)
    for t in (zr, zs, zt, zp):
        ttnn.deallocate(t)

    # ---- A3: what one pass over the same bytes costs -------------------------------------------
    print("=== A3 single-pass floor ===", flush=True)
    a3 = {"z_MB": round(I * C * D * J * 2 / 1e6, 1)}
    a3["clone_dram2dram_us"] = us(timed(dev, lambda: ttnn.deallocate(
        ttnn.clone(z, memory_config=DRAM))))
    try:
        a3["transpose_2d_us"] = us(timed(dev, lambda: ttnn.deallocate(ttnn.permute(z, (1, 0)))))
    except Exception as e:                                              # noqa: BLE001
        a3["transpose_2d_us"] = str(e)[:100]
    a3["one_pass_ms_per_fold"] = round(a3["clone_dram2dram_us"] * K / 1000, 1)
    res["A3_single_pass"] = a3
    print("  " + json.dumps(a3), flush=True)

    # ---- A1: b-side reorder. Same matmul shapes, different output index meaning ------------------
    print("=== A1 b-side reorder (N = j*D+d) ===", flush=True)
    a1 = {}
    a1["matmul_us"] = us(timed(dev, lambda: ttnn.deallocate(
        ttnn.matmul(a_flat, b_jd, transpose_b=True, compute_kernel_config=ckc))))
    a1["matmul_delta_pct"] = round(100 * (a1["matmul_us"] - a0["matmul_3037"]) / a0["matmul_3037"], 1)
    z2 = ttnn.matmul(a_flat, b_jd, transpose_b=True, compute_kernel_config=ckc)
    # (i*C+c, j*D+d) -> (i, j, c*D+d). Both splits are on 32-multiples, so both are tile-grid
    # regroupings; the permute of the two middle axes is the only real motion.
    for tag, fn in (("reshape4d", lambda: ttnn.reshape(z2, (I, C, J, D))),):
        try:
            a1[f"{tag}_us"] = us(timed(dev, lambda: ttnn.deallocate(fn())))
        except Exception as e:                                          # noqa: BLE001
            a1[f"{tag}_us"] = str(e)[:110]
    try:
        z4 = ttnn.reshape(z2, (I, C, J, D))
        a1["permute_0213_us"] = us(timed(dev, lambda: ttnn.deallocate(
            ttnn.permute(z4, (0, 2, 1, 3)))))
        zz = ttnn.permute(z4, (0, 2, 1, 3))
        a1["final_reshape_us"] = us(timed(dev, lambda: ttnn.deallocate(
            ttnn.reshape(zz, (I, J, C * D)))))
        a1["shape_after"] = list(zz.shape)
        ttnn.deallocate(zz)
        ttnn.deallocate(z4)
    except Exception as e:                                              # noqa: BLE001
        a1["permute_0213_us"] = str(e)[:140]
    ttnn.deallocate(z2)
    res["A1_b_reorder"] = a1
    print("  " + json.dumps(a1), flush=True)

    # ---- A2: batched over i --------------------------------------------------------------------
    print("=== A2 batched over i ===", flush=True)
    a2 = {}
    try:
        a2["matmul_us"] = us(timed(dev, lambda: ttnn.deallocate(
            ttnn.matmul(a, b_dj, transpose_b=True, compute_kernel_config=ckc)), 1, 2, 3))
        a2["ratio_vs_A0"] = round(a2["matmul_us"] / a0["matmul_3037"], 2)
        a2["delta_ms_per_fold"] = round((a2["matmul_us"] - a0["matmul_3037"]) * K / 1000, 1)
    except Exception as e:                                              # noqa: BLE001
        a2["matmul_us"] = str(e)[:140]
    res["A2_batched_over_i"] = a2
    print("  " + json.dumps(a2), flush=True)

    # ---- core ladder on the two ops that expose a grid knob ------------------------------------
    print("=== core ladder ===", flush=True)
    lad = {}
    zt2 = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT), (I, C * D, J)),
                         ttnn.TILE_LAYOUT)
    zp2 = ttnn.permute(zt2, (0, 2, 1))
    for gx, gy in ((4, 4), (8, 4), (11, 8), (11, 10)):
        try:
            g = ttnn.CoreGrid(x=gx, y=gy)
            lad[f"linear_3068_{gx}x{gy}"] = us(timed(dev, lambda g=g: ttnn.deallocate(
                ttnn.linear(zp2, ow, compute_kernel_config=ckc, core_grid=g)), 1, 2, 3))
        except Exception as e:                                          # noqa: BLE001
            lad[f"linear_3068_{gx}x{gy}"] = str(e)[:80]
        print(f"  {gx}x{gy}: {lad[f'linear_3068_{gx}x{gy}']}", flush=True)
    res["core_ladder_us"] = lad

    (args.out / "p5_opm_chain.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
