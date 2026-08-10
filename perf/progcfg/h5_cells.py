#!/usr/bin/env python3
"""H5 — is the pair-track L1-output projection's win program config, traffic, or lost overlap?

Four cells, not two. `_PAIR_PROJ_L1_OUT` holds the program config fixed and moves only the output
buffer, so the production A/B cannot separate the two terms; crossing them can:

                    output DRAM                 output L1
    untuned         C0D  core_grid=             C0L
    tuned           C1D  = production OFF       C1L = production ON

    config term at DRAM        C0D - C1D
    destination term, tuned    C1D - C1L        <- the whole production win
    interaction                (C0D-C0L) - (C1D-C1L)

Everything is priced at the fold's OWN shapes, `[1, N, N, c] @ [c, c]` bf16 with the source in DRAM,
which is where `TriangleMultiplication`'s `ttnn.layer_norm` leaves it. `[1,320,320,32]` overstated two
retracted projections in this org by 3.2x and 1.45x and is not used.

Three walls per cell, because the 2.30 ms/call the sibling leg reported is a REGION cost and not a
per-projection one (`_trimul_out_proj` runs twice per trimul, `tenstorrent.py:1751` and `1754`):

    proj             the projection alone
    region           p_out, g_out, multiply_(sigmoid) -- `tenstorrent.py:1751-1759`
    region_consume   the region plus a downstream add_ reading the result

Roofs are measured in this process on this card and nothing is inherited (charter 4.1). Both sides of
every timed region synchronise: an unsynced drain has inverted rankings in this codebase before.

    env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-progcfg-h5 \
        TT_MESH_GRAPH_DESC_PATH=<ttnn>/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
        PYTHONPATH=<worktree> python3 perf/progcfg/h5_cells.py --out perf/progcfg/h5_cells_qb2c0.json
"""
import argparse, json, statistics as st, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch    # noqa: E402
import ttnn     # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def ceil32(v):
    return -(-v // 32) * 32


def timed(dev, fn, warm=3, pipe=4, reps=5):
    """Median seconds per call. Synchronise on both sides of every timed region."""
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


def shard_cfg(rows, cols, gy, gx):
    cr = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(cr, [rows // gy, cols // gx], ttnn.ShardOrientation.ROW_MAJOR))


def cfg_fields(pc):
    """Every field of a program config, so 'the two arms are identical' is printed, not assumed."""
    if pc is None:
        return None
    keys = ("compute_with_storage_grid_size", "in0_block_w", "out_subblock_h", "out_subblock_w",
            "out_block_h", "out_block_w", "per_core_M", "per_core_N", "fuse_batch", "mcast_in0")
    d = {}
    for k in keys:
        try:
            v = getattr(pc, k)
        except Exception:
            continue
        d[k] = str(v) if not isinstance(v, (int, bool, float)) else v
    return d


# ---------------------------------------------------------------------------------------------
# roofs
# ---------------------------------------------------------------------------------------------

def roofs(dev, gx, gy):
    """DRAM read / write / read+write, the L1 op roof, and the square compute roof. This card only."""
    out = {}
    n = 4096                                       # 4096x4096 bf16 = 33.55 MB, 4 of them = 134 MB
    big = 8192                                     # 8192x8192 bf16 = 134.2 MB, the 512 aa pair size
    t = torch.randn(big, big, dtype=torch.bfloat16)
    d_src = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    nbytes = big * big * 2

    # DRAM -> L1 : DRAM sees reads only.  L1 is 160 MB across 110 banks so 134 MB fits, just.
    try:
        f = lambda: ttnn.deallocate(ttnn.clone(d_src, memory_config=L1))          # noqa: E731
        s = timed(dev, f, warm=2, pipe=2, reps=5)
        out["dram_read_GBps"] = nbytes / s / 1e9
        out["dram_read_ms"] = s * 1e3
    except Exception as e:
        out["dram_read_error"] = str(e)[:300]

    # L1 -> DRAM : DRAM sees writes only.
    try:
        l1_src = ttnn.clone(d_src, memory_config=L1)
        f = lambda: ttnn.deallocate(ttnn.clone(l1_src, memory_config=DRAM))       # noqa: E731
        s = timed(dev, f, warm=2, pipe=2, reps=5)
        out["dram_write_GBps"] = nbytes / s / 1e9
        ttnn.deallocate(l1_src)
    except Exception as e:
        out["dram_write_error"] = str(e)[:300]

    # DRAM -> DRAM : both directions, the roof a region moving reads and writes is scored against.
    f = lambda: ttnn.deallocate(ttnn.clone(d_src, memory_config=DRAM))            # noqa: E731
    s = timed(dev, f, warm=2, pipe=2, reps=5)
    out["dram_rw_GBps"] = 2 * nbytes / s / 1e9
    ttnn.deallocate(d_src)

    # L1 op roof: block-sharded bf16 add over the whole grid, 3N-byte convention (2 in, 1 out).
    rows, cols = 32 * gy * 8, 32 * gx * 8
    sc = shard_cfg(rows, cols, gy, gx)
    a = ttnn.from_torch(torch.randn(rows, cols, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=sc)
    b = ttnn.from_torch(torch.randn(rows, cols, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=sc)
    f = lambda: ttnn.deallocate(ttnn.add(a, b, memory_config=sc))                 # noqa: E731
    s = timed(dev, f)
    out["l1_op_GBps"] = 3 * rows * cols * 2 / s / 1e9
    ttnn.deallocate(a)
    ttnn.deallocate(b)

    # Square compute roof, bf16 HiFi4, DRAM output. A FLOOR on the square roof, not the roof:
    # the curve was still rising at 4096 when the sibling leg swept it.
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    a = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    b = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    f = lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc,     # noqa: E731
                                            memory_config=DRAM, dtype=ttnn.bfloat16))
    s = timed(dev, f, warm=2, pipe=2, reps=5)
    out["compute_square_TFLOPs"] = 2 * n ** 3 / s / 1e12
    ttnn.deallocate(a)
    ttnn.deallocate(b)

    out["machine_balance_FLOP_per_byte"] = (
        out["compute_square_TFLOPs"] * 1e12 / (out.get("dram_read_GBps", 0) * 1e9)
        if out.get("dram_read_GBps") else None)
    return out


# ---------------------------------------------------------------------------------------------
# the cells
# ---------------------------------------------------------------------------------------------

def manual_cfg(gx, gy, bw, obh, obw, pcm, pcn):
    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
    sw = max(w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
        out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
        per_core_M=pcm, per_core_N=pcn, fuse_batch=True, fused_activation=None, mcast_in0=False)


def run_shape(dev, T, N, c, gx, gy, ckc, ref_check=True):
    """Every cell at one fold shape. Returns a dict; a cell that throws records its exception."""
    import tt_bio.tenstorrent as tt

    ncores = gx * gy
    m_tiles = N * (ceil32(N) // 32)
    k_tiles = n_tiles = ceil32(c) // 32
    per_core_M = -(-(-(-m_tiles // ncores)) // 5) * 5
    cores_engaged = -(-m_tiles // per_core_M)
    tile, bank = 2048, tt._l1_bank_bytes()

    res = {"N": N, "c": c, "m_tiles": m_tiles, "k_tiles": k_tiles, "n_tiles": n_tiles,
           "per_core_M": per_core_M, "cores_engaged": cores_engaged, "grid_cores": ncores,
           "l1_bank_bytes": bank, "cells": {}, "sweep": {}}

    # bytes and FLOPs at the padded shape ttnn actually moves
    rows = N * ceil32(N)
    tbytes = rows * ceil32(c) * 2
    res["tensor_bytes"] = tbytes
    res["proj_flops"] = 2 * rows * ceil32(c) * ceil32(c)
    res["arithmetic_intensity"] = res["proj_flops"] / (2 * tbytes)      # in + out, weights negligible
    res["region_bytes_6T"] = 6 * tbytes

    xt = torch.randn(1, N, N, c, dtype=torch.bfloat16)
    x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    x2 = ttnn.from_torch(torch.randn(1, N, N, c, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    wp = ttnn.from_torch(torch.randn(c, c, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    wg = ttnn.from_torch(torch.randn(c, c, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)

    # the copy roof AT THIS SHAPE -- the only fair roof for an op whose output dominates its traffic
    for tag, mc in (("dram", DRAM), ("l1", L1)):
        try:
            f = lambda mc=mc: ttnn.deallocate(ttnn.clone(x, memory_config=mc))    # noqa: E731
            s = timed(dev, f, warm=2, pipe=2, reps=5)
            res[f"clone_{tag}_ms"] = s * 1e3
            res[f"clone_{tag}_GBps"] = 2 * tbytes / s / 1e9
        except Exception as e:
            res[f"clone_{tag}_error"] = str(e)[:200]

    tuned_dram = tt._pair_proj_config(x, wp, bw_cap=tt._PAIR_PROJ_BW, out_l1=False)
    tuned_l1 = tt._pair_proj_config(x, wp, bw_cap=tt._PAIR_PROJ_L1_BW, out_l1=True)
    res["cfg_tuned_dram"] = cfg_fields(tuned_dram)
    res["cfg_tuned_l1"] = cfg_fields(tuned_l1)
    res["cfg_fields_identical"] = (res["cfg_tuned_dram"] == res["cfg_tuned_l1"]
                                   if tuned_l1 is not None else None)

    def linear(src, w, pc, mc):
        if pc is None:
            return ttnn.linear(src, w, memory_config=mc, dtype=ttnn.bfloat16,
                               compute_kernel_config=ckc, core_grid=tt.CORE_GRID_MAIN)
        return ttnn.linear(src, w, memory_config=mc, dtype=ttnn.bfloat16,
                           compute_kernel_config=ckc, program_config=pc)

    ref = None

    def cell(name, pc, mc, untuned=False):
        entry = {"program_config": cfg_fields(pc) if not untuned else "core_grid=CORE_GRID_MAIN",
                 "output": "L1" if mc is L1 else "DRAM"}
        pcc = None if untuned else pc
        try:
            def proj():
                ttnn.deallocate(linear(x, wp, pcc, mc))
            entry["proj_ms"] = timed(dev, proj, pipe=2) * 1e3

            def region():
                p = linear(x, wp, pcc, mc)
                g = linear(x2, wg, pcc, mc)
                r = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                ttnn.deallocate(g)
                ttnn.deallocate(r)
            entry["region_ms"] = timed(dev, region, pipe=2) * 1e3

            def region_consume():
                p = linear(x, wp, pcc, mc)
                g = linear(x2, wg, pcc, mc)
                r = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                ttnn.deallocate(g)
                z = ttnn.add(x2, r, memory_config=DRAM)
                ttnn.deallocate(r)
                ttnn.deallocate(z)
            entry["region_consume_ms"] = timed(dev, region_consume, pipe=2) * 1e3

            if ref_check:
                o = linear(x, wp, pcc, mc)
                h = ttnn.to_torch(o)
                ttnn.deallocate(o)
                nonlocal ref
                if ref is None:
                    ref = h
                    entry["torch_equal_vs_C1D"] = True
                else:
                    entry["torch_equal_vs_C1D"] = bool(torch.equal(h, ref))
                    entry["max_abs_vs_C1D"] = float((h.float() - ref.float()).abs().max())
        except Exception as e:                                                    # noqa: BLE001
            entry["error"] = str(e)[:400]
            entry["traceback"] = traceback.format_exc()[-600:]
        res["cells"][name] = entry

    # C1D first so it is the torch.equal reference: a memory config cannot change a value, so any
    # cell that differs from it has changed the CONTRACTION ORDER and is a different parity class.
    cell("C1D_tuned_dram", tuned_dram, DRAM)
    cell("C1L_tuned_l1", tuned_l1, L1)
    cell("C0D_untuned_dram", None, DRAM, untuned=True)
    cell("C0L_untuned_l1", None, L1, untuned=True)

    # --- the DRAM-output config sweep: can a config recover the destination's win? (P6) ----------
    divs = lambda v: [d for d in range(1, v + 1) if v % d == 0]                    # noqa: E731
    for bw in [d for d in divs(k_tiles) if d <= 16]:
        for obh in [d for d in divs(per_core_M) if d <= 25]:
            for obw in divs(n_tiles):
                need = (2 * bw * (obh + obw) * tile + obh * obw * (tile + 4096) + 128 * 1024)
                if need > bank:
                    continue
                key = f"bw{bw}_obh{obh}_obw{obw}"
                try:
                    pc = manual_cfg(gx, gy, bw, obh, obw, per_core_M, n_tiles)

                    def proj(pc=pc):
                        ttnn.deallocate(linear(x, wp, pc, DRAM))
                    res["sweep"][key] = {"need_bytes": need, "proj_ms": timed(dev, proj, pipe=2) * 1e3}
                except Exception as e:                                             # noqa: BLE001
                    res["sweep"][key] = {"need_bytes": need, "error": str(e)[:200]}

    # --- P7: an L1 OUTPUT config the production path never tries, because it holds bw and obw ----
    res["l1_out_alternatives"] = {}
    for bw in [d for d in divs(k_tiles) if d <= 16]:
        for obh in [d for d in divs(per_core_M) if d <= 25]:
            for obw in divs(n_tiles):
                need = (2 * bw * (obh + obw) * tile + obh * obw * (tile + 4096) + 128 * 1024
                        + per_core_M * n_tiles * tile)
                if need > bank:
                    continue
                key = f"bw{bw}_obh{obh}_obw{obw}"
                try:
                    pc = manual_cfg(gx, gy, bw, obh, obw, per_core_M, n_tiles)

                    def proj(pc=pc):
                        ttnn.deallocate(linear(x, wp, pc, L1))

                    def region(pc=pc):
                        p = linear(x, wp, pc, L1)
                        g = linear(x2, wg, pc, L1)
                        r = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                        ttnn.deallocate(g)
                        ttnn.deallocate(r)
                    e = {"need_bytes": need, "pct_of_bank": 100.0 * need / bank,
                         "proj_ms": timed(dev, proj, pipe=2) * 1e3,
                         "region_ms": timed(dev, region, pipe=2) * 1e3}
                    o = linear(x, wp, pc, L1)
                    h = ttnn.to_torch(o)
                    ttnn.deallocate(o)
                    e["torch_equal_vs_C1D"] = bool(torch.equal(h, ref)) if ref is not None else None
                    res["l1_out_alternatives"][key] = e
                except Exception as ex:                                            # noqa: BLE001
                    res["l1_out_alternatives"][key] = {
                        "need_bytes": need, "pct_of_bank": 100.0 * need / bank,
                        "error": str(ex)[:300]}

    for t_ in (x, x2, wp, wg):
        try:
            ttnn.deallocate(t_)
        except Exception:
            pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="298:64,512:64,298:256,512:256",
                    help="N:c pairs, the fold's own shapes")
    ap.add_argument("--skip-roofs", action="store_true")
    a = ap.parse_args()

    import importlib.metadata as im
    from tt_bio.tenstorrent import get_device, COMPUTE_GRID_MAIN
    import tt_bio.tenstorrent as tt

    dev = get_device()
    gx, gy = COMPUTE_GRID_MAIN
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    out = {
        "host": "qb2", "chip": 0, "ttnn": im.version("ttnn"),
        "device_grid": list(dev.compute_with_storage_grid_size()),
        "compute_grid_main": [gx, gy],
        "max_worker_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
        "l1_bank_bytes": tt._l1_bank_bytes(),
        "flags": {"_PAIR_PROJ_BW": tt._PAIR_PROJ_BW, "_PAIR_PROJ_L1_BW": tt._PAIR_PROJ_L1_BW,
                  "_NARROW_PROJ_BW": tt._NARROW_PROJ_BW, "_PAIR_PROJ_L1_OUT": tt._PAIR_PROJ_L1_OUT},
        "note": "qb2 at 0.68.0: every figure here is a RATIO owing a qb1/0.67.4 re-take (charter 4.8)",
        "shapes": {},
    }
    if not a.skip_roofs:
        out["roofs"] = roofs(dev, gx, gy)
        print(json.dumps(out["roofs"], indent=2), flush=True)

    for spec in a.shapes.split(","):
        N, c = (int(v) for v in spec.split(":"))
        print(f"--- shape N={N} c={c} ---", flush=True)
        try:
            out["shapes"][spec] = run_shape(dev, torch, N, c, gx, gy, ckc)
        except Exception as e:                                                     # noqa: BLE001
            out["shapes"][spec] = {"error": str(e)[:400], "traceback": traceback.format_exc()[-800:]}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(out["shapes"][spec], indent=2)[:3000], flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
