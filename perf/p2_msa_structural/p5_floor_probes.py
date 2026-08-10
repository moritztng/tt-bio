#!/usr/bin/env python3
"""P5 Phase-2 probes: roofs on THIS card, C3 (PWA weight slices) and C4 (the transition floor).

Everything here is a probe. Nothing under `tt_bio/` is touched.

  roofs : DRAM read / write / copy ladder, the per-op launch floor on this chip, and the two
          K-corrected compute rates my stages actually run at (K=1024 nt=8 for the OPM consumer,
          K=64 nt=2 for the template pair track).
  c3    : does `ttnn.slice` cost scale with the slice size? 16 kB / 160 kB / 1.6 MB off ONE tensor,
          plus an alignment A/B at fixed output bytes that separates "per-launch floor" from
          "sub-tile slice".
  c4    : row-count sweep at 1x / 2x / 4x on the transition layer_norm + linear, and an output-width
          sweep on `minimal_matmul` (the @1720 / @1726 pair, T5's @1695 / @1701).

    TT_VISIBLE_DEVICES=1 TT_MESH_GRAPH_DESC_PATH=... PYTHONPATH=$PWD \
      python3 perf/p2_msa_structural/p5_floor_probes.py roofs c3 c4 --out perf/p2_msa_structural/
"""
from __future__ import annotations

import argparse, json, statistics as st, sys, time
from pathlib import Path

import torch, ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=4, pipe=5, reps=7):
    """Median of `reps` regions of `pipe` back-to-back calls, synced on BOTH sides of every region."""
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


def dealloc(t):
    ttnn.deallocate(t)


def roofs(dev, ckc, res):
    print("=== DRAM ladder ===", flush=True)
    ladder = []
    for mb in (16, 32, 64, 96, 128):
        nrow = int(mb * 1e6 / 2) // 4096
        nb = nrow * 4096 * 2
        r = {"MB": round(nb / 1e6, 2)}
        try:
            xd = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            r["read_GBs"] = round(nb / timed(dev, lambda: dealloc(ttnn.clone(xd, memory_config=L1)),
                                             2, 3, 5) / 1e9, 1)
            r["dram2dram_GBs"] = round(2 * nb / timed(
                dev, lambda: dealloc(ttnn.clone(xd, memory_config=DRAM)), 2, 3, 5) / 1e9, 1)
            xl = ttnn.clone(xd, memory_config=L1)
            r["write_GBs"] = round(nb / timed(dev, lambda: dealloc(ttnn.clone(xl, memory_config=DRAM)),
                                              2, 3, 5) / 1e9, 1)
            dealloc(xl); dealloc(xd)
        except Exception as e:                                          # noqa: BLE001
            r["err"] = str(e)[:120]
        ladder.append(r); print("  " + json.dumps(r), flush=True)
    res["dram"] = {"ladder": ladder,
                   "read_peak_GBs": max((r.get("read_GBs", 0) for r in ladder), default=None),
                   "write_peak_GBs": max((r.get("write_GBs", 0) for r in ladder), default=None),
                   "copy_peak_GBs": max((r.get("dram2dram_GBs", 0) for r in ladder), default=None)}

    print("=== per-op launch floor on this chip ===", flush=True)
    floor = {}
    for tiles in (1, 2, 8, 32):
        x = ttnn.from_torch(torch.randn(32, 32 * tiles), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
        floor[f"relu_{tiles}t"] = round(timed(dev, lambda: dealloc(ttnn.relu(x, memory_config=L1)),
                                              8, 20, 7) * 1e6, 2)
        floor[f"clone_{tiles}t"] = round(timed(dev, lambda: dealloc(ttnn.clone(x, memory_config=L1)),
                                               8, 20, 7) * 1e6, 2)
        dealloc(x)
    res["launch_floor_us"] = floor
    print("  " + json.dumps(floor), flush=True)

    print("=== square compute roof + the two K-corrected rates my stages run ===", flush=True)
    comp = {}
    for n in (4096, 6144):
        a = ttnn.from_torch(torch.randn(n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)
        b = ttnn.from_torch(torch.randn(n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)
        s = timed(dev, lambda: dealloc(ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                   memory_config=DRAM, dtype=ttnn.bfloat16)), 2, 3, 5)
        comp[f"square_{n}"] = {"us": round(s * 1e6, 1), "tflops": round(2 * n ** 3 / s / 1e12, 2)}
        dealloc(a); dealloc(b)
        print(f"  square {n}: {comp[f'square_{n}']}", flush=True)
    for tag, (batch, M, K, N) in (("K1024_nt8_opm_consumer", (298, 320, 1024, 256)),
                                  ("K64_nt2_template", (298, 320, 64, 64)),
                                  ("K256_nt1_pwa_zbias", (298, 320, 256, 32))):
        try:
            a = ttnn.from_torch(torch.randn(1, batch, M, K) * .1, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            w = ttnn.from_torch(torch.randn(K, N) * .1, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            s = timed(dev, lambda: dealloc(ttnn.linear(a, w, compute_kernel_config=ckc,
                                                       core_grid=CORE_GRID_MAIN)), 2, 3, 5)
            comp[tag] = {"us": round(s * 1e6, 1),
                         "tflops": round(2 * batch * M * K * N / s / 1e12, 2),
                         "out_write_GBs": round(batch * M * N * 2 / s / 1e9, 1)}
            dealloc(a); dealloc(w)
        except Exception as e:                                          # noqa: BLE001
            comp[tag] = {"err": str(e)[:120]}
        print(f"  {tag}: {comp[tag]}", flush=True)
    res["compute"] = comp


def c3(dev, ckc, res):
    """C3 kill test: is `ttnn.slice` per-launch bound or bandwidth bound, and is the tile face it?"""
    print("=== C3: slice size sweep off ONE tensor ===", flush=True)
    W = ttnn.from_torch(torch.randn(4096, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=DRAM)
    out = {}
    for tag, (r, c) in (("16kB", (256, 32)), ("160kB", (256, 320)), ("1.6MB", (256, 3200))):
        s = timed(dev, lambda r=r, c=c: dealloc(W[0:r, 0:c]), 4, 10, 7)
        out[tag] = {"us": round(s * 1e6, 2), "out_kB": round(r * c * 2 / 1e3, 1),
                    "GBs": round(2 * r * c * 2 / s / 1e9, 3)}
        print(f"  {tag}: {out[tag]}", flush=True)
    res["c3_size_sweep"] = out

    print("=== C3: alignment A/B at fixed rows ===", flush=True)
    al = {}
    for tag, (r0, r1, c0, c1) in (("col_0_1_subtile", (0, 256, 0, 1)),
                                  ("col_1_2_subtile", (0, 256, 1, 2)),
                                  ("col_0_32_aligned", (0, 256, 0, 32)),
                                  ("col_1_33_unaligned", (0, 256, 1, 33)),
                                  ("col_32_64_aligned", (0, 256, 32, 64))):
        s = timed(dev, lambda a=r0, b=r1, c=c0, d=c1: dealloc(W[a:b, c:d]), 4, 10, 7)
        al[tag] = {"us": round(s * 1e6, 2), "out_kB": round((r1 - r0) * (c1 - c0) * 2 / 1e3, 2)}
        print(f"  {tag}: {al[tag]}", flush=True)
    res["c3_alignment"] = al
    dealloc(W)

    print("=== C3: the production slices at their real shapes ===", flush=True)
    # PairWeightedAveraging, 8 heads, head_dim 8, c_z=256, c_m=128. torch_to_tt transposes the
    # torch (out,in) weight to (in,out), so the sliced axis is the OUTPUT axis in every case.
    prod = {}
    for tag, (shape, sl) in (
            ("z_weight[:, i:i+1]",   ((256, 8), lambda t: t[:, 1:2])),
            ("m_weight[:, i*8:+8]",  ((128, 64), lambda t: t[:, 8:16])),
            ("g_weight[:, i*8:+8]",  ((128, 64), lambda t: t[:, 8:16])),
            ("o_weight[i*8:+8, :]",  ((64, 128), lambda t: t[8:16, :]))):
        t = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)
        s = timed(dev, lambda t=t, sl=sl: dealloc(sl(t)), 4, 10, 7)
        prod[tag] = {"src_shape": list(shape), "us": round(s * 1e6, 2)}
        print(f"  {tag}: {prod[tag]}", flush=True)
        dealloc(t)
    res["c3_production"] = prod


def c4(dev, ckc, res):
    """C4 kill test: flat in rows confirms a fixed per-op cost; linear in rows kills it."""
    print("=== C4: row-count sweep, transition layer_norm + linear (L1 resident) ===", flush=True)
    c_m, hidden = 128, 512
    lnw = ttnn.from_torch(torch.randn(c_m), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    lnb = ttnn.from_torch(torch.randn(c_m), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w1 = ttnn.from_torch(torch.randn(c_m, hidden) * .1, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    rows = {}
    for mult in (1, 2, 4):
        d = 35 * mult
        x = ttnn.from_torch(torch.randn(d, 320, c_m) * .1, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
        r = {"depth": d, "in_MB": round(d * 320 * c_m * 2 / 1e6, 2)}
        r["layer_norm_us"] = round(timed(dev, lambda: dealloc(ttnn.layer_norm(
            x, weight=lnw, bias=lnb, epsilon=1e-5, compute_kernel_config=ckc,
            memory_config=L1)), 4, 8, 7) * 1e6, 2)
        r["linear_silu_us"] = round(timed(dev, lambda: dealloc(ttnn.linear(
            x, w1, activation="silu", compute_kernel_config=ckc, memory_config=L1,
            core_grid=CORE_GRID_MAIN)), 4, 8, 7) * 1e6, 2)
        r["linear_plain_us"] = round(timed(dev, lambda: dealloc(ttnn.linear(
            x, w1, compute_kernel_config=ckc, memory_config=L1,
            core_grid=CORE_GRID_MAIN)), 4, 8, 7) * 1e6, 2)
        rows[f"{mult}x"] = r
        print(f"  {mult}x: {r}", flush=True)
        dealloc(x)
    res["c4_rows"] = rows

    print("=== C4: minimal_matmul output-width sweep (T5's @1695 vs @1701) ===", flush=True)
    x = ttnn.from_torch(torch.randn(320, 320, 64) * .1, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    mm = {}
    for N in (32, 64, 128, 192, 384):
        w = ttnn.from_torch(torch.randn(64, N) * .1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)
        try:
            s = timed(dev, lambda: dealloc(ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)),
                3, 5, 7)
            mm[f"N{N}"] = {"us": round(s * 1e6, 2), "out_MB": round(320 * 320 * N * 2 / 1e6, 2),
                           "write_GBs": round(320 * 320 * N * 2 / s / 1e9, 1),
                           "tflops": round(2 * 320 * 320 * 64 * N / s / 1e12, 2)}
        except Exception as e:                                          # noqa: BLE001
            mm[f"N{N}"] = {"err": str(e)[:120]}
        print(f"  N={N}: {mm[f'N{N}']}", flush=True)
        dealloc(w)
    dealloc(x)
    res["c4_minimal_matmul"] = mm

    print("=== C4: core ladder on the 1x transition linear (>20 us op) ===", flush=True)
    x = ttnn.from_torch(torch.randn(35, 320, c_m) * .1, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    lad = {}
    for gx, gy in ((1, 1), (4, 4), (8, 4), (11, 8), (11, 10)):
        try:
            g = ttnn.CoreGrid(x=gx, y=gy)
            lad[f"{gx}x{gy}"] = round(timed(dev, lambda g=g: dealloc(ttnn.linear(
                x, w1, compute_kernel_config=ckc, memory_config=L1, core_grid=g)), 3, 6, 5) * 1e6, 2)
        except Exception as e:                                          # noqa: BLE001
            lad[f"{gx}x{gy}"] = str(e)[:80]
        print(f"  {gx}x{gy}: {lad[f'{gx}x{gy}']}", flush=True)
    res["c4_core_ladder_us"] = lad
    dealloc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=["roofs", "c3", "c4"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    dg = dev.compute_with_storage_grid_size()
    res = {"device": {"host": "qb2", "card": 1, "compute_grid": f"{dg.x}x{dg.y}",
                      "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                      "ttnn": "0.68.0",
                      "ckc": "HiFi4 fp32_dest_acc_en=True packer_l1_acc=True"}}
    print(json.dumps(res["device"]), flush=True)
    for w in args.which:
        {"roofs": roofs, "c3": c3, "c4": c4}[w](dev, ckc, res)
        (args.out / "p5_floor_probes.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
