#!/usr/bin/env python3
"""p3-l1-output — the L1-output candidate, measured at the PRODUCTION grid.

X2 measured `projection + add` with program configs built from a module-scope `COMPUTE_GRID_MAIN`
= 11x10 while the fold runs 13x10 (X2's correction 8). Everything here reads the grid AFTER the
device is open, so these are the first figures for this candidate at the grid production uses.

Three arms:

  pair    X2's six legs, reproduced: projection alone and `projection + add`, DRAM vs L1 output,
          untuned `core_grid` vs the tuned 1D config, with `torch.equal` against the core_grid
          DRAM leg.
  trimul  the sequence production actually runs at `tenstorrent.py:1619/1622`:
          proj(p_out) -> proj(g_out) -> multiply_(p,g,sigmoid) -> add_(z, .). Four calls of the
          class per Pairformer layer share two residual adds, so the probe's pair over-counts the
          add half; this arm measures the real chain instead of pricing per call.
  site2   deliverable 2, `tenstorrent.py:2088`: layer_norm(z) -> linear(z,[256,16]) -> permute.
          The projection reads 48.82 MB and writes 6.10 MB, so the lever under test is the SOURCE
          (an L1 layer_norm output) and not the output.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tt_bio.tenstorrent as T  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TOK, NPAD, C_Z, N_HEADS = 298, 320, 256, 16


def timed(fn, dev, warm=3, pipe=3, reps=7):
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


def cfg(gx, gy, m_tiles, k_tiles, n_tiles, bw, obh, out_l1):
    """The production `_pair_proj_program_config` formula, with the OUTPUT term added when the
    output lands in L1 (`perfwar-programconfig-gate-output-not-subtracted`: the shipped helper
    omits it because it always writes to DRAM)."""
    nc = gx * gy
    if m_tiles < nc or k_tiles % bw:
        return None
    pcm = -(-(-(-m_tiles // nc)) // obh) * obh
    if pcm > m_tiles or -(-m_tiles // pcm) > nc:
        return None
    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
    sw = max(w for w in range(min(4 // sh, n_tiles), 0, -1) if n_tiles % w == 0)
    need = 2 * bw * (obh + n_tiles) * 2048 + obh * n_tiles * (2048 + 4096) + 128 * 1024
    if out_l1:
        need += pcm * n_tiles * 2048
    if need > T._l1_bank_bytes():
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=obh, out_block_w=n_tiles, per_core_M=pcm,
        per_core_N=n_tiles, fuse_batch=True, fused_activation=None, mcast_in0=False)


def l1_free(dev):
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    return int(mv.largest_contiguous_bytes_free_per_bank)


# ---------------------------------------------------------------------------------- arm: pair ---
def arm_pair(dev, ckc, gx, gy, res):
    x = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    z = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    w = ttnn.from_torch(torch.randn(C_Z, C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=DRAM)
    m_tiles, k_tiles, n_tiles = TOK * (NPAD // 32), C_Z // 32, C_Z // 32
    out_bytes = m_tiles * 32 * C_Z * 2
    banks = gx * gy
    res["pair_meta"] = {"m_tiles": m_tiles, "output_bytes": out_bytes,
                        "output_kB_per_bank": round(out_bytes / banks / 1024, 1),
                        "l1_bank_bytes": T._l1_bank_bytes(), "banks": banks,
                        "l1_free_before": l1_free(dev)}
    legs, outs = {}, {}
    for lbl, omem, pc in (
            ("dram_cg", DRAM, None),
            ("dram_tuned_bw8_obh5", DRAM, cfg(gx, gy, m_tiles, k_tiles, n_tiles, 8, 5, False)),
            ("dram_tuned_bw1_obh5", DRAM, cfg(gx, gy, m_tiles, k_tiles, n_tiles, 1, 5, False)),
            ("l1_cg", L1, None),
            ("l1_tuned_bw1_obh5", L1, cfg(gx, gy, m_tiles, k_tiles, n_tiles, 1, 5, True)),
            ("l1_tuned_bw8_obh5", L1, cfg(gx, gy, m_tiles, k_tiles, n_tiles, 8, 5, True)),
            ("l1_tuned_bw8_obh2", L1, cfg(gx, gy, m_tiles, k_tiles, n_tiles, 8, 2, True)),
            ("l1_tuned_bw8_obh1", L1, cfg(gx, gy, m_tiles, k_tiles, n_tiles, 8, 1, True))):
        kw = dict(memory_config=omem, dtype=ttnn.bfloat16, compute_kernel_config=ckc)
        if pc is None and lbl.endswith("cg"):
            proj = lambda: ttnn.linear(x, w, core_grid=T.CORE_GRID_MAIN, **kw)   # noqa: E731
        elif pc is None:
            legs[lbl] = {"err": "config refused by the L1 budget"}
            print(f"  {lbl:22s} REFUSED by the L1 budget", flush=True)
            continue
        else:
            proj = (lambda c: lambda: ttnn.linear(x, w, program_config=c, **kw))(pc)

        def pair():
            p = proj()
            o = ttnn.add(z, p, memory_config=DRAM)
            ttnn.deallocate(p)
            ttnn.deallocate(o)

        row = {}
        try:
            row["proj_us"] = round(timed(lambda: ttnn.deallocate(proj()), dev) * 1e6, 2)
            row["proj_write_GBs"] = round(out_bytes / (row["proj_us"] / 1e6) / 1e9, 1)
            row["pair_us"] = round(timed(pair, dev) * 1e6, 2)
            p = proj()
            ttnn.synchronize_device(dev)
            row["l1_free_while_output_live"] = l1_free(dev)
            outs[lbl] = ttnn.to_torch(ttnn.add(z, p, memory_config=DRAM))
            ttnn.deallocate(p)
        except Exception as e:                                                # noqa: BLE001
            row["err"] = str(e)[:200]
        legs[lbl] = row
        print(f"  {lbl:22s} proj {row.get('proj_us','-'):>9} us  pair {row.get('pair_us','-'):>9} us"
              f"  L1free {row.get('l1_free_while_output_live','-')}  {row.get('err','')}", flush=True)

    ref = outs.get("dram_cg")
    for lbl, row in legs.items():
        o = outs.get(lbl)
        if ref is not None and o is not None:
            row["torch_equal_vs_dram_cg"] = bool(torch.equal(ref, o))
            row["max_abs_vs_dram_cg"] = float((ref.double() - o.double()).abs().max())
    res["pair"] = legs
    for t in (x, z, w):
        ttnn.deallocate(t)


# -------------------------------------------------------------------------------- arm: trimul ---
def arm_trimul(dev, ckc, gx, gy, res):
    """proj(p_out) -> proj(g_out) -> multiply_(p,g,sigmoid) -> add_(z, .), as production runs it."""
    xn = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z) * 0.1, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    xg = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z) * 0.1, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    wp = ttnn.from_torch(torch.randn(C_Z, C_Z) * 0.05, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    wg = ttnn.from_torch(torch.randn(C_Z, C_Z) * 0.05, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    z0 = torch.randn(1, TOK, TOK, C_Z) * 0.1
    m_tiles, k_tiles, n_tiles = TOK * (NPAD // 32), C_Z // 32, C_Z // 32

    def mk(bw, obh, out_l1):
        return cfg(gx, gy, m_tiles, k_tiles, n_tiles, bw, obh, out_l1)

    legs, outs = {}, {}
    #  label                      p_out mem, p cfg,            g_out mem, g cfg
    plan = (("prod_today",        DRAM, mk(8, 5, False),  DRAM, mk(8, 5, False)),
            ("all_dram_bw1",      DRAM, mk(1, 5, False),  DRAM, mk(1, 5, False)),
            ("p_l1_bw1",          L1,   mk(1, 5, True),   DRAM, mk(1, 5, False)),
            ("p_l1_bw8_obh5",     L1,   mk(8, 5, True),   DRAM, mk(8, 5, False)),
            ("p_l1_bw8_obh2",     L1,   mk(8, 2, True),   DRAM, mk(8, 2, False)),
            ("both_l1_bw1",       L1,   mk(1, 5, True),   L1,   mk(1, 5, True)),
            ("both_l1_bw8_obh2",  L1,   mk(8, 2, True),   L1,   mk(8, 2, True)))
    for lbl, pmem, pcfg, gmem, gcfg in plan:
        if pcfg is None or gcfg is None:
            legs[lbl] = {"err": "config refused by the L1 budget"}
            print(f"  {lbl:20s} REFUSED by the L1 budget", flush=True)
            continue
        z = ttnn.from_torch(z0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)

        def chain():
            p = ttnn.linear(xn, wp, memory_config=pmem, dtype=ttnn.bfloat16,
                            compute_kernel_config=ckc, program_config=pcfg)
            g = ttnn.linear(xg, wg, memory_config=gmem, dtype=ttnn.bfloat16,
                            compute_kernel_config=ckc, program_config=gcfg)
            o = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g)
            ttnn.add_(z, o)
            ttnn.deallocate(o)

        row = {}
        try:
            row["chain_us"] = round(timed(chain, dev, warm=2, pipe=2, reps=7) * 1e6, 2)
            p = ttnn.linear(xn, wp, memory_config=pmem, dtype=ttnn.bfloat16,
                            compute_kernel_config=ckc, program_config=pcfg)
            ttnn.synchronize_device(dev)
            row["l1_free_after_p"] = l1_free(dev)
            g = ttnn.linear(xg, wg, memory_config=gmem, dtype=ttnn.bfloat16,
                            compute_kernel_config=ckc, program_config=gcfg)
            ttnn.synchronize_device(dev)
            row["l1_free_after_p_and_g"] = l1_free(dev)
            o = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g)
            zc = ttnn.from_torch(z0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=dev, memory_config=DRAM)
            ttnn.add_(zc, o)
            outs[lbl] = ttnn.to_torch(zc)
            ttnn.deallocate(o)
            ttnn.deallocate(zc)
        except Exception as e:                                                # noqa: BLE001
            row["err"] = str(e)[:200]
        ttnn.deallocate(z)
        legs[lbl] = row
        print(f"  {lbl:20s} chain {row.get('chain_us','-'):>9} us  L1free after p "
              f"{row.get('l1_free_after_p','-')}  after p+g {row.get('l1_free_after_p_and_g','-')}"
              f"  {row.get('err','')}", flush=True)

    ref = outs.get("all_dram_bw1")
    for lbl, row in legs.items():
        o = outs.get(lbl)
        if ref is not None and o is not None:
            row["torch_equal_vs_all_dram_bw1"] = bool(torch.equal(ref, o))
            row["max_abs_vs_all_dram_bw1"] = float((ref.double() - o.double()).abs().max())
    res["trimul"] = legs
    for t in (xn, xg, wp, wg):
        ttnn.deallocate(t)


# --------------------------------------------------------------------------------- arm: site2 ---
def arm_site2(dev, ckc, gx, gy, res):
    """`tenstorrent.py:2088` — layer_norm(z) -> linear(z, [256,16]) -> permute(0,3,1,2)."""
    z = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    nw = ttnn.from_torch(torch.randn(C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=dev, memory_config=DRAM)
    nb = ttnn.from_torch(torch.randn(C_Z) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=dev, memory_config=DRAM)
    w = ttnn.from_torch(torch.randn(C_Z, N_HEADS), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=DRAM)
    m_tiles, k_tiles, n_tiles = TOK * (NPAD // 32), C_Z // 32, 1
    res["site2_meta"] = {"in0_bytes": TOK * NPAD * C_Z * 2, "out_bytes": TOK * NPAD * 32 * 2}

    legs, outs = {}, {}
    #  label              norm mem, proj cfg (bw, obh),   proj out mem
    plan = (("prod_bw1",        DRAM, (1, 5), DRAM),
            ("prod_cg",         DRAM, None,   DRAM),
            ("bw1_outL1",       DRAM, (1, 5), L1),
            ("normL1_bw1",      L1,   (1, 5), DRAM),
            ("normL1_bw1_outL1", L1,  (1, 5), L1),
            ("normL1_cg",       L1,   None,   DRAM),
            ("bw8_outDRAM",     DRAM, (8, 5), DRAM),
            ("normL1_bw8",      L1,   (8, 5), DRAM))
    for lbl, nmem, bwobh, omem in plan:
        pc = None
        if bwobh is not None:
            pc = cfg(gx, gy, m_tiles, k_tiles, n_tiles, bwobh[0], bwobh[1], omem is L1)
            if pc is None:
                legs[lbl] = {"err": "config refused by the L1 budget"}
                print(f"  {lbl:20s} REFUSED by the L1 budget", flush=True)
                continue

        def region():
            zn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5,
                                 compute_kernel_config=ckc, memory_config=nmem)
            if pc is None:
                zb = ttnn.linear(zn, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN,
                                 memory_config=omem, dtype=ttnn.bfloat16)
            else:
                zb = ttnn.linear(zn, w, compute_kernel_config=ckc, program_config=pc,
                                 memory_config=omem, dtype=ttnn.bfloat16)
            ttnn.deallocate(zn)
            zp = ttnn.permute(zb, (0, 3, 1, 2))
            ttnn.deallocate(zb)
            ttnn.deallocate(zp)

        def parts():
            zn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5,
                                 compute_kernel_config=ckc, memory_config=nmem)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            if pc is None:
                zb = ttnn.linear(zn, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN,
                                 memory_config=omem, dtype=ttnn.bfloat16)
            else:
                zb = ttnn.linear(zn, w, compute_kernel_config=ckc, program_config=pc,
                                 memory_config=omem, dtype=ttnn.bfloat16)
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            ttnn.deallocate(zn)
            zp = ttnn.permute(zb, (0, 3, 1, 2))
            ttnn.synchronize_device(dev)
            t2 = time.perf_counter()
            ttnn.deallocate(zb)
            return zp, (t1 - t0), (t2 - t1)

        row = {}
        try:
            row["region_us"] = round(timed(region, dev, warm=2, pipe=2, reps=7) * 1e6, 2)
            pj, pm = [], []
            for _ in range(5):
                zp, a, b = parts()
                pj.append(a * 1e6)
                pm.append(b * 1e6)
                ttnn.deallocate(zp)
            row["proj_us"] = round(st.median(pj), 2)
            row["permute_us"] = round(st.median(pm), 2)
            zp, _, _ = parts()
            outs[lbl] = ttnn.to_torch(zp)
            ttnn.deallocate(zp)
        except Exception as e:                                                # noqa: BLE001
            row["err"] = str(e)[:200]
        legs[lbl] = row
        print(f"  {lbl:20s} region {row.get('region_us','-'):>9} us  proj {row.get('proj_us','-'):>9}"
              f"  permute {row.get('permute_us','-'):>8}  {row.get('err','')}", flush=True)

    ref = outs.get("prod_cg")
    for lbl, row in legs.items():
        o = outs.get(lbl)
        if ref is not None and o is not None:
            row["torch_equal_vs_prod_cg"] = bool(torch.equal(ref, o))
            row["max_abs_vs_prod_cg"] = float((ref.double() - o.double()).abs().max())
    res["site2"] = legs
    for t in (z, nw, nb, w):
        ttnn.deallocate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    choices=["pair", "trimul", "site2"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    # Read the grid AFTER the device is open: `_apply_grid_thresholds` rebinds COMPUTE_GRID_MAIN
    # from 11x10 to 13x10 at device open, and a probe that imported it at module scope builds
    # program configs for a grid production does not use (X2's correction 8).
    gx, gy = T.COMPUTE_GRID_MAIN
    dg = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    res = {"card": "qb1 card 1", "ttnn": getattr(ttnn, "__version__", "?"),
           "compute_grid_main_after_device_open": [gx, gy],
           "device_compute_with_storage_grid": [dg.x, dg.y],
           "core_grid_main": f"{T.CORE_GRID_MAIN.x}x{T.CORE_GRID_MAIN.y}",
           "l1_bank_bytes": T._l1_bank_bytes()}
    print(json.dumps({k: v for k, v in res.items()}), flush=True)
    for arm in a.arm:
        print(f"=== arm {arm} ===", flush=True)
        {"pair": arm_pair, "trimul": arm_trimul, "site2": arm_site2}[arm](dev, ckc, gx, gy, res)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
