#!/usr/bin/env python3
"""r4t-bwd micro-screen + float64 VJP + broken-arm checks on Blackhole (B2 dW, B3 dX, B4a/b/c).

Measures every RFD4 crop-256 PairUpdate/Trunk backward shape at AICLK >= 1200 MHz (sampled during):
  - B2 dW (K=65536, bf16->fp32 & fp32->fp32, incl. ragged N=24/12): stock vs widened _split_k vs
    minimal_matmul (default and swept MinimalMatmulConfig) + SPLIT_K_BROKEN negative control.
  - B3 dX (M=65536, bf16->bf16): stock vs minimal_matmul (default and swept MinimalMatmulConfig) +
    MMVJP_BROKEN negative control.
  - B4a _sum_leading (65536x384, 65536x768 bf16): stock vs split-row + SUM_LEADING_BROKEN control.
  - B4b lnbw CENTER=0 RMSNorm (pair 256x256x384, token 256x768 w/ and w/o gamma, 256x256): composed
    vs fused lnbw kernel + float64 rel_l2.
  - B4c widen_add vs typecast+add (256x256x384 fp32+bf16): speed + bit-exact check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def sample_aiclk(card: int, stop: threading.Event, out: list):
    path = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
    while not stop.is_set():
        try:
            out.append(int(open(path).read().split()[0]))
        except Exception:
            pass
        time.sleep(0.05)


def time_op(dev, fn, ttnn, warm: int = 2, reps: int = 5):
    for _ in range(warm):
        o = fn()
        ttnn.synchronize_device(dev)
        if isinstance(o, tuple):
            for x in o:
                if x is not None:
                    ttnn.deallocate(x)
        elif o is not None:
            ttnn.deallocate(o)
    ts = []
    last = None
    for i in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1000.0)
        if i == reps - 1:
            last = o
        else:
            if isinstance(o, tuple):
                for x in o:
                    if x is not None:
                        ttnn.deallocate(x)
            elif o is not None:
                ttnn.deallocate(o)
    ts.sort()
    return ts[len(ts) // 2], ts, last


def rel_l2(got_tt, ref64, ttnn, torch):
    g = ttnn.to_torch(got_tt).double().reshape(ref64.shape)
    return float((g - ref64).norm() / ref64.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import autograd as ag, lnbw, rne_add, tenstorrent as TT

    dev = TT.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    grid = dev.compute_with_storage_grid_size()
    cfg = ag.precise_config()
    sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()
    clk, stop = [], threading.Event()
    thr = threading.Thread(target=sample_aiclk, args=(card, stop, clk), daemon=True)
    thr.start()

    res = {
        "host": os.uname().nodename,
        "chip": card,
        "grid": [int(grid.x), int(grid.y)],
        "sha": sha,
        "reps": a.reps,
        "dw": [],
        "dx": [],
        "sum_leading": [],
        "rms_bw": [],
        "widen_add": {},
        "broken": {},
    }
    gen = torch.Generator().manual_seed(17)
    K = 65536

    # ---- B2: dW shapes ----
    dw_shapes = [
        (256, 384, "bf16"),
        (384, 384, "bf16"),
        (384, 1024, "bf16"),
        (768, 384, "bf16"),
        (384, 768, "bf16"),
        (384, 24, "bf16"),
        (384, 12, "bf16"),
        (384, 128, "fp32"),
    ]
    print("=== B2 dW SCREEN ===", flush=True)
    for M, N, dtn in dw_shapes:
        tdt = torch.float32 if dtn == "fp32" else torch.bfloat16
        ttdt = ttnn.float32 if dtn == "fp32" else ttnn.bfloat16
        X = torch.randn(K, M, generator=gen, dtype=torch.float32).to(tdt)
        G = torch.randn(K, N, generator=gen, dtype=torch.float32).to(tdt)
        ref = X.double().T @ G.double()
        tx = ttnn.from_torch(X, dtype=ttdt, layout=ttnn.TILE_LAYOUT, device=dev)
        tg = ttnn.from_torch(G, dtype=ttdt, layout=ttnn.TILE_LAYOUT, device=dev)

        # Stock ttnn.matmul (bypass _split_k)
        ms_stock, _, o_stock = time_op(
            dev, lambda: ttnn.matmul(tx, tg, transpose_a=True, compute_kernel_config=cfg, dtype=ttnn.float32),
            ttnn, reps=a.reps)
        r_stock = rel_l2(o_stock, ref, ttnn, torch)
        ttnn.deallocate(o_stock)

        row = {"M": M, "N": N, "K": K, "dtype": dtn, "stock_ms": round(ms_stock, 3), "stock_rel": round(r_stock, 6),
               "split_k": {}, "min_mm": {}}

        # Widened _split_k sweep over P in (16, 32, 64)
        old_out, old_ch, old_brk = ag.SPLIT_K_OUT_TILES, ag.SPLIT_K_CHUNKS, ag.SPLIT_K_BROKEN
        ag.SPLIT_K_OUT_TILES, ag.SPLIT_K_BROKEN = 144, False
        for P in (16, 32, 64):
            ag.SPLIT_K_CHUNKS = P
            try:
                ms_sk, _, o_sk = time_op(
                    dev, lambda: ag._split_k(tx, tg, {"transpose_a": True, "compute_kernel_config": cfg, "dtype": ttnn.float32}),
                    ttnn, reps=a.reps)
                if o_sk is not None:
                    r_sk = rel_l2(o_sk, ref, ttnn, torch)
                    ttnn.deallocate(o_sk)
                    row["split_k"][f"P{P}"] = {"ms": round(ms_sk, 3), "rel": round(r_sk, 6)}
            except Exception as e:
                row["split_k"][f"P{P}"] = {"err": str(e)[:120]}
        # Broken arm on first eligible shape
        if M == 256 and N == 384:
            ag.SPLIT_K_CHUNKS, ag.SPLIT_K_BROKEN = 64, True
            o_brk = ag._split_k(tx, tg, {"transpose_a": True, "compute_kernel_config": cfg, "dtype": ttnn.float32})
            r_brk = rel_l2(o_brk, ref, ttnn, torch)
            ttnn.deallocate(o_brk)
            res["broken"]["split_k_256x384"] = {"rel": round(r_brk, 6), "fail": r_brk > 0.1}
        ag.SPLIT_K_OUT_TILES, ag.SPLIT_K_CHUNKS, ag.SPLIT_K_BROKEN = old_out, old_ch, old_brk

        # minimal_matmul (bf16 tile-multiple only)
        if dtn == "bf16" and M % 32 == 0 and N % 32 == 0:
            ms_t, _, xt = time_op(dev, lambda: ttnn.transpose(tx, -2, -1), ttnn, reps=a.reps)
            row["transpose_x_ms"] = round(ms_t, 3)
            # Default minimal_matmul
            ms_def, _, o_def = time_op(
                dev, lambda: ttnn.experimental.minimal_matmul(xt, tg, compute_kernel_config=cfg, dtype=ttnn.float32),
                ttnn, reps=a.reps)
            r_def = rel_l2(o_def, ref, ttnn, torch)
            ttnn.deallocate(o_def)
            row["min_mm"]["default"] = {"mm_ms": round(ms_def, 3), "tot_ms": round(ms_def + ms_t, 3), "rel": round(r_def, 6)}

            # Swept MinimalMatmulConfig candidates
            Mt, Nt = M // 32, N // 32
            cands = []
            for mb in (1, 2, 4):
                if Mt % mb:
                    continue
                for nb in (1, 2, 4):
                    if Nt % nb:
                        continue
                    blocks = (Mt // mb) * (Nt // nb)
                    if blocks < 32 or blocks > 220:
                        continue
                    sh = 2 if mb % 2 == 0 else 1
                    sw = min(4 // sh, nb) if nb % min(4 // sh, nb) == 0 else 1
                    for kb in (4, 8, 16):
                        cands.append((mb, kb, nb, sh, sw))
            for mb, kb, nb, sh, sw in cands:
                tag = f"m{mb}_k{kb}_n{nb}_s{sh}x{sw}"
                mc = ttnn.MinimalMatmulConfig(
                    M_block_size=mb, K_block_size=kb, N_block_size=nb,
                    subblock_h=sh, subblock_w=sw, compute_with_storage_grid_size=grid)
                try:
                    ms_c, _, o_c = time_op(
                        dev, lambda mc=mc: ttnn.experimental.minimal_matmul(
                            xt, tg, compute_kernel_config=cfg, dtype=ttnn.float32, config=mc),
                        ttnn, reps=a.reps)
                    r_c = rel_l2(o_c, ref, ttnn, torch)
                    ttnn.deallocate(o_c)
                    row["min_mm"][tag] = {"mm_ms": round(ms_c, 3), "tot_ms": round(ms_c + ms_t, 3), "rel": round(r_c, 6)}
                except Exception as e:
                    row["min_mm"][tag] = {"err": str(e)[:80]}
            ttnn.deallocate(xt)

        for t in (tx, tg):
            ttnn.deallocate(t)
        res["dw"].append(row)
        best_mm = min(((v["mm_ms"], k) for k, v in row["min_mm"].items() if "mm_ms" in v), default=(None, None))
        best_sk = min(((v["ms"], k) for k, v in row["split_k"].items() if "ms" in v), default=(None, None))
        print(f"dW {M:3d}x{N:4d} ({dtn}): stock={ms_stock:.3f}ms (rel {r_stock:.5f}) | "
              f"best_sk={best_sk} | best_mm={best_mm} (T={row.get('transpose_x_ms')})", flush=True)

    # ---- B3: dX shapes ----
    print("=== B3 dX SCREEN ===", flush=True)
    dx_shapes = [(1024, 384), (384, 384), (768, 384), (384, 768), (384, 256)]
    for Cin, Cout in dx_shapes:
        G = torch.randn(K, Cin, generator=gen, dtype=torch.float32).bfloat16()
        W = (torch.randn(Cout, Cin, generator=gen, dtype=torch.float32) / (Cout ** 0.5)).bfloat16()
        ref = G.double() @ W.double().T
        tg = ttnn.from_torch(G, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tw = ttnn.from_torch(W, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        ms_stock, _, o_stock = time_op(
            dev, lambda: ttnn.matmul(tg, tw, transpose_b=True, compute_kernel_config=cfg), ttnn, reps=a.reps)
        r_stock = rel_l2(o_stock, ref, ttnn, torch)
        ttnn.deallocate(o_stock)

        row = {"M": K, "K": Cin, "N": Cout, "stock_ms": round(ms_stock, 3), "stock_rel": round(r_stock, 6), "min_mm": {}}
        wt = ttnn.transpose(tw, -2, -1)
        ms_def, _, o_def = time_op(
            dev, lambda: ttnn.experimental.minimal_matmul(tg, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16),
            ttnn, reps=a.reps)
        r_def = rel_l2(o_def, ref, ttnn, torch)
        ttnn.deallocate(o_def)
        row["min_mm"]["default"] = {"ms": round(ms_def, 3), "rel": round(r_def, 6)}

        Nt, Kt = Cout // 32, Cin // 32
        for mb, kb, nb, sh, sw in (
            (8, 4, 2, 2, 2), (8, 8, 2, 2, 2), (8, 4, 4, 1, 4), (8, 8, 4, 1, 4),
            (4, 4, 2, 2, 2), (4, 8, 4, 1, 4), (16, 4, 2, 2, 2), (16, 8, 4, 1, 4),
        ):
            if Kt % kb or Nt % nb:
                continue
            tag = f"m{mb}_k{kb}_n{nb}_s{sh}x{sw}"
            mc = ttnn.MinimalMatmulConfig(
                M_block_size=mb, K_block_size=kb, N_block_size=nb,
                subblock_h=sh, subblock_w=sw, compute_with_storage_grid_size=grid)
            try:
                ms_c, _, o_c = time_op(
                    dev, lambda mc=mc: ttnn.experimental.minimal_matmul(
                        tg, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16, config=mc),
                    ttnn, reps=a.reps)
                r_c = rel_l2(o_c, ref, ttnn, torch)
                ttnn.deallocate(o_c)
                row["min_mm"][tag] = {"ms": round(ms_c, 3), "rel": round(r_c, 6)}
            except Exception as e:
                row["min_mm"][tag] = {"err": str(e)[:80]}

        if Cin == 384 and Cout == 384:
            old_m, old_b = ag.MMVJP_MIN, ag.MMVJP_BROKEN
            ag.MMVJP_MIN, ag.MMVJP_BROKEN = True, True
            o_brk = ag._mmvjp_dx(tg, tw, cfg)
            r_brk = rel_l2(o_brk, ref, ttnn, torch)
            ttnn.deallocate(o_brk)
            res["broken"]["mmvjp_dx_384x384"] = {"rel": round(r_brk, 6), "fail": r_brk > 0.5}
            ag.MMVJP_MIN, ag.MMVJP_BROKEN = old_m, old_b

        for t in (tg, tw, wt):
            ttnn.deallocate(t)
        res["dx"].append(row)
        best_mm = min(((v["ms"], k) for k, v in row["min_mm"].items() if "ms" in v), default=(None, None))
        print(f"dX 65536x{Cin:4d}->{Cout:3d}: stock={ms_stock:.3f}ms (rel {r_stock:.5f}) | best_mm={best_mm}", flush=True)

    # ---- B4a: _sum_leading ----
    print("=== B4a _sum_leading SCREEN ===", flush=True)
    for C in (384, 768):
        G = torch.randn(K, C, generator=gen, dtype=torch.float32).bfloat16()
        ref = G.double().sum(dim=0, keepdim=True)
        tg = ttnn.from_torch(G, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        old_sp, old_ch, old_brk = ag.SUM_LEADING_SPLIT, ag.SUM_LEADING_CHUNKS, ag.SUM_LEADING_BROKEN
        ag.SUM_LEADING_SPLIT = False
        ms_st, _, o_st = time_op(dev, lambda: ag._sum_leading(tg, [1, C]), ttnn, reps=a.reps)
        r_st = rel_l2(o_st, ref, ttnn, torch)
        ttnn.deallocate(o_st)
        row = {"N": K, "C": C, "stock_ms": round(ms_st, 3), "stock_rel": round(r_st, 6), "split": {}}
        ag.SUM_LEADING_SPLIT, ag.SUM_LEADING_BROKEN = True, False
        for P in (16, 32, 64):
            ag.SUM_LEADING_CHUNKS = P
            ms_sp, _, o_sp = time_op(dev, lambda: ag._sum_leading(tg, [1, C]), ttnn, reps=a.reps)
            r_sp = rel_l2(o_sp, ref, ttnn, torch)
            ttnn.deallocate(o_sp)
            row["split"][f"P{P}"] = {"ms": round(ms_sp, 3), "rel": round(r_sp, 6)}
        if C == 384:
            ag.SUM_LEADING_CHUNKS, ag.SUM_LEADING_BROKEN = 32, True
            o_brk = ag._sum_leading(tg, [1, C])
            r_brk = rel_l2(o_brk, ref, ttnn, torch)
            ttnn.deallocate(o_brk)
            res["broken"]["sum_leading_384"] = {"rel": round(r_brk, 6), "fail": r_brk > 0.1}
        ag.SUM_LEADING_SPLIT, ag.SUM_LEADING_CHUNKS, ag.SUM_LEADING_BROKEN = old_sp, old_ch, old_brk
        ttnn.deallocate(tg)
        res["sum_leading"].append(row)
        print(f"sum_leading 65536x{C}: stock={ms_st:.3f}ms (rel {r_st:.5f}) | split={row['split']}", flush=True)

    # ---- B4b: lnbw CENTER=0 RMSNorm ----
    print("=== B4b RMSNorm bw SCREEN ===", flush=True)
    rms_cases = [
        ("pair_384_now", (1, 256, 256, 384), False),
        ("tok_768_w", (1, 256, 768), True),
        ("tok_768_now", (1, 256, 768), False),
        ("tok_256_now", (1, 256, 256), False),
    ]
    old_rms = lnbw.RMS
    for nm, sh, has_w in rms_cases:
        X = torch.randn(*sh, generator=gen).bfloat16()
        G = torch.randn(*sh, generator=gen).bfloat16()
        W = (torch.randn(1, sh[-1], generator=gen) * 0.5 + 1.0).bfloat16() if has_w else None
        xr = X.double().requires_grad_(True)
        wr = W.double().requires_grad_(True) if has_w else None
        y = xr * torch.rsqrt((xr * xr).mean(-1, keepdim=True) + 1e-5)
        if has_w:
            y = y * wr
        (y * G.double()).sum().backward()
        tx = ttnn.from_torch(X, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tg = ttnn.from_torch(G, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tw = ttnn.from_torch(W, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) if has_w else None

        def run_composed():
            lnbw.RMS = False
            x_t = ag.Tensor(tx, requires_grad=True)
            w_t = ag.Tensor(tw, requires_grad=True) if has_w else None
            ag._layer_norm_bw(x_t, w_t, None, 1e-5, cfg, center=False)(tg)
            return (x_t.grad, w_t.grad if has_w else None)

        def run_fused():
            lnbw.RMS = True
            if has_w:
                dx, dg, _ = lnbw.layer_norm_bw_weights(tx, tg, tw, 1e-5, cfg, center=False, do_beta=False)
                return (dx, dg)
            return (lnbw.layer_norm_bw(tx, tg, None, 1e-5, center=False), None)

        ms_c, _, oc = time_op(dev, run_composed, ttnn, reps=a.reps)
        rc_x = rel_l2(oc[0], xr.grad, ttnn, torch)
        rc_w = rel_l2(oc[1], wr.grad, ttnn, torch) if has_w else None
        for o in oc:
            if o is not None:
                ttnn.deallocate(o)

        ms_f, _, of = time_op(dev, run_fused, ttnn, reps=a.reps)
        rf_x = rel_l2(of[0], xr.grad, ttnn, torch)
        rf_w = rel_l2(of[1], wr.grad, ttnn, torch) if has_w else None
        for o in of:
            if o is not None:
                ttnn.deallocate(o)

        for t in (tx, tg, tw):
            if t is not None:
                ttnn.deallocate(t)
        row = {"case": nm, "shape": list(sh), "has_w": has_w,
               "composed_ms": round(ms_c, 3), "fused_ms": round(ms_f, 3),
               "composed_dx_rel": round(rc_x, 6), "fused_dx_rel": round(rf_x, 6),
               "composed_dw_rel": None if rc_w is None else round(rc_w, 6),
               "fused_dw_rel": None if rf_w is None else round(rf_w, 6)}
        res["rms_bw"].append(row)
        print(f"rms_bw {nm:14s}: composed={ms_c:.3f}ms -> fused={ms_f:.3f}ms | "
              f"dx_rel {rc_x:.5f}->{rf_x:.5f} dw_rel {rc_w}->{rf_w}", flush=True)
    lnbw.RMS = old_rms

    # ---- B4c: widen_add vs typecast+add ----
    print("=== B4c widen_add SCREEN ===", flush=True)
    A = torch.randn(1, 256, 256, 384, generator=gen, dtype=torch.float32)
    B = torch.randn(1, 256, 256, 384, generator=gen, dtype=torch.float32).bfloat16()
    ta = ttnn.from_torch(A, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(B, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def run_tc_add():
        b32 = ttnn.typecast(tb, ttnn.float32)
        out = ttnn.add(ta, b32)
        ttnn.deallocate(b32)
        return out

    old_wa = rne_add.WIDEN_ADD
    rne_add.WIDEN_ADD = True
    ms_tc, _, o_tc = time_op(dev, run_tc_add, ttnn, reps=a.reps)
    ms_wa, _, o_wa = time_op(dev, lambda: rne_add.widen_add(ta, tb), ttnn, reps=a.reps)
    t_tc = ttnn.to_torch(o_tc).float()
    t_wa = ttnn.to_torch(o_wa).float()
    max_diff = float((t_tc - t_wa).abs().max())
    bit_exact = bool(torch.equal(t_tc, t_wa))
    res["widen_add"] = {
        "shape": [1, 256, 256, 384],
        "tc_add_ms": round(ms_tc, 3),
        "widen_add_ms": round(ms_wa, 3),
        "saved_ms": round(ms_tc - ms_wa, 3),
        "max_abs_diff": max_diff,
        "bit_exact": bit_exact,
        "digest": hashlib.sha256(t_wa.numpy().tobytes()).hexdigest()[:16],
    }
    print(f"widen_add (256,256,384): tc+add={ms_tc:.3f}ms -> widen_add={ms_wa:.3f}ms "
          f"bit_exact={bit_exact} max_diff={max_diff:.3e}", flush=True)
    rne_add.WIDEN_ADD = old_wa
    for t in (ta, tb, o_tc, o_wa):
        ttnn.deallocate(t)

    stop.set()
    thr.join(timeout=2)
    res["aiclk"] = [min(clk or [0]), max(clk or [0])]
    res["aiclk_med"] = sorted(clk)[len(clk) // 2] if clk else 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"DONE screen_mm aiclk={res['aiclk']} med={res['aiclk_med']} broken={res['broken']}", flush=True)


if __name__ == "__main__":
    main()
