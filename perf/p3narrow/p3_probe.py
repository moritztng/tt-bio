#!/usr/bin/env python3
"""p3-narrow-write probes, qb1 card 1.

Arms
  wroof  the card's matmul-writer write roof by OUTPUT WIDTH (P2: use `minimal_matmul`, it wins
         every width below 64, so a `ttnn.linear` roof at a narrow output is 1.37x low)
  sites  the two narrow-output sites at the fold's own [1,298,320,256], three config arms, with
         torch.equal / RMSD / PCC of each arm against the production baseline
  l1out  deliverable 3 candidate 1: a pair-track projection with an L1 output, standalone and
         under a live-block L1 occupancy, plus free L1 per bank at the call site
  fuse   deliverable 3 candidate 2: TriangleAttention's qkv + g + triangle_bias are three matmuls
         on the SAME layer-normed x. Fused into one nt=33 output vs the three separate calls,
         with the slice-back cost that the consumers need priced separately.

Every timed region synchronises immediately before the clock starts and before it stops.
"""
import argparse, json, math, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tt_bio.tenstorrent import (  # noqa: E402
    CORE_GRID_MAIN, COMPUTE_GRID_MAIN, get_device, _l1_bank_bytes,
)

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
_BANKS = 130
TOK, NPAD, C_Z = 298, 320, 256


def timed(fn, dev, warm=4, pipe=4, reps=7):
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


def l1_free_per_bank(dev):
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    return int(mv.largest_contiguous_bytes_free_per_bank), int(mv.total_bytes_free_per_bank)


def pair_cfg(m_tiles, k_tiles, n_tiles, bw, obh, out_l1=False):
    """The shipped `_pair_proj_program_config` family with in0_block_w / out_block_h exposed.

    L1 budget carries the output block's bf16 tile AND the fp32 partial the packer accumulates
    into (perfwar-programconfig-gate-output-not-subtracted).
    """
    gx, gy = COMPUTE_GRID_MAIN
    ncores = gx * gy
    if m_tiles < ncores or k_tiles % bw:
        return None
    per_core_M = -(-(-(-m_tiles // ncores)) // obh) * obh
    if per_core_M > m_tiles or -(-m_tiles // per_core_M) > ncores:
        return None
    obw = n_tiles
    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
    sw = max(w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0)
    tile = 2048
    need = 2 * bw * (obh + obw) * tile + obh * obw * (tile + 4096) + 128 * 1024
    if out_l1:
        need += per_core_M * obw * tile
    if need > _l1_bank_bytes():
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
        out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
        per_core_M=per_core_M, per_core_N=n_tiles, fuse_batch=True,
        fused_activation=None, mcast_in0=False,
    )


def cores_for(m_tiles, obh):
    gx, gy = COMPUTE_GRID_MAIN
    ncores = gx * gy
    per_core_M = -(-(-(-m_tiles // ncores)) // obh) * obh
    return -(-m_tiles // per_core_M), per_core_M


def stats(ref_t, got):
    """torch.equal + RMSD + max abs + PCC. Accumulated in float64: a float32 reduction over
    24.4 M elements returned PCC 1.0003 for a pair of bitwise-identical tensors."""
    a = ref_t.to(torch.float64).flatten()
    b = ttnn.to_torch(got).to(torch.float64).flatten()
    eq = bool(torch.equal(ref_t, ttnn.to_torch(got)))
    d = a - b
    rmsd = float(torch.sqrt((d * d).mean()))
    mx = float(d.abs().max())
    am, bm = a - a.mean(), b - b.mean()
    pcc = float((am * bm).sum() / (am.norm() * bm.norm() + 1e-30))
    return {"torch_equal": eq, "rmsd": rmsd, "max_abs": mx, "pcc": pcc,
            "rel_rmsd_vs_std": rmsd / float(a.std())}


# --------------------------------------------------------------------------------------------
def arm_wroof(dev, ckc, res):
    """Best write rate this card's matmul writer reaches, by output width nt. K=256, L1 operands."""
    rows = []
    for nt in (1, 2, 8, 16, 32, 64, 128):
        n = nt * 32
        # hold the output near 48-52 MB so every width writes a comparable byte count
        # hold the output near 52 MB, but never let the L1 in0 (M x 256 bf16) exceed 40 MB:
        # at nt=1 a 52 MB output would need a 417 MB in0, which no L1 holds. Output size is
        # recorded per row -- an L1-operand roof that does not name its size is not reproducible.
        m = min(78080, max(1024, int(52e6 / 2 / n) // 32 * 32))
        a = ttnn.from_torch(torch.randn(m, C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=L1)
        w = ttnn.from_torch(torch.randn(C_Z, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=L1)
        wr = m * n * 2
        best = None
        legs = [("minimal_matmul", lambda: ttnn.experimental.minimal_matmul(
                    a, w, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc)),
                ("linear_core_grid", lambda: ttnn.linear(
                    a, w, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                    core_grid=CORE_GRID_MAIN))]
        for bw in (1, 8):
            for obh in (5, 2):
                cfg = pair_cfg(m // 32, 8, nt, bw, obh)
                if cfg is not None:
                    legs.append((f"1d_bw{bw}_obh{obh}", (lambda c: lambda: ttnn.linear(
                        a, w, memory_config=DRAM, dtype=ttnn.bfloat16,
                        compute_kernel_config=ckc, program_config=c))(cfg)))
        for lbl, fn in legs:
            try:
                s = timed(lambda: ttnn.deallocate(fn()), dev)
            except Exception as e:                                        # noqa: BLE001
                rows.append({"nt": nt, "leg": lbl, "err": str(e)[:90]})
                continue
            gbs = wr / s / 1e9
            tf = 2 * m * C_Z * n / s / 1e12
            rows.append({"nt": nt, "leg": lbl, "M": m, "out_MB": round(wr / 1e6, 2),
                         "in0_MB": round(m * C_Z * 2 / 1e6, 2), "us": round(s * 1e6, 2),
                         "write_GBs": round(gbs, 1), "tflops": round(tf, 2)})
            if best is None or gbs > best[1]:
                best = (lbl, gbs, round(s * 1e6, 2), round(tf, 2))
            print(f"  nt={nt:<4} {lbl:18s} {s*1e6:9.2f} us  {gbs:7.1f} GB/s  {tf:7.2f} TFLOP/s",
                  flush=True)
        if best is not None:
            res.setdefault("wroof_best", {})[str(nt)] = {
                "leg": best[0], "write_GBs": round(best[1], 1), "us": best[2], "tflops": best[3]}
        ttnn.deallocate(a)
        ttnn.deallocate(w)
    res["wroof"] = rows


def _pair_x(dev):
    """The fold's own pair tensor: logical [1,298,298,256], padded [1,298,320,256], DRAM."""
    t = torch.randn(1, TOK, TOK, C_Z)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=DRAM)


def arm_sites(dev, ckc, res):
    x = _pair_x(dev)
    print(f"  x padded={list(x.padded_shape)} logical={list(x.shape)} "
          f"buf={str(x.memory_config().buffer_type).split('.')[-1]}", flush=True)
    res["in0_buffer_type"] = str(x.memory_config().buffer_type).split(".")[-1]
    res["in0_padded_shape"] = list(x.padded_shape)
    m_tiles = TOK * (NPAD // 32)
    out = {}
    for site, ncol in (("pwa_z_bias", 1), ("template_z_proj", 64)):
        w = ttnn.from_torch(torch.randn(C_Z, ncol), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)
        nt = max(1, -(-ncol // 32))
        legs = [("baseline_core_grid", None, None)]
        for bw, obh in ((1, 5), (8, 5)):
            legs.append((f"bw{bw}_obh{obh}", bw, obh))
        ref = None
        rows = []
        for lbl, bw, obh in legs:
            if bw is None:
                fn = lambda: ttnn.linear(x, w, memory_config=DRAM, dtype=ttnn.bfloat16,
                                         compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
                cores, pcm = None, None         # core_grid derives its own split; see the ladder
            else:
                cfg = pair_cfg(m_tiles, 8, nt, bw, obh)
                if cfg is None:
                    rows.append({"leg": lbl, "err": "config refused"})
                    continue
                fn = (lambda c: lambda: ttnn.linear(
                    x, w, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                    program_config=c))(cfg)
                cores, pcm = cores_for(m_tiles, obh)
            s = timed(fn, dev)
            o = fn()
            ttnn.synchronize_device(dev)
            if ref is None:
                ref = ttnn.to_torch(o)
                par = {"torch_equal": True, "rmsd": 0.0, "max_abs": 0.0, "pcc": 1.0,
                       "rel_rmsd_vs_std": 0.0}
            else:
                par = stats(ref, o)
            ttnn.deallocate(o)
            wr = m_tiles * 32 * nt * 32 * 2
            rows.append({"leg": lbl, "us": round(s * 1e6, 2), "cores": cores, "per_core_M": pcm,
                         "write_GBs": round(wr / s / 1e9, 1),
                         "read_GBs": round(m_tiles * 32 * C_Z * 2 / s / 1e9, 1),
                         "tflops": round(2 * m_tiles * 32 * C_Z * nt * 32 / s / 1e12, 2),
                         **par})
            print(f"  {site:16s} {lbl:20s} {s*1e6:9.2f} us cores={str(cores):<6} "
                  f"eq={par['torch_equal']} pcc={par['pcc']:.8f} rmsd={par['rmsd']:.4e}",
                  flush=True)
        # a core_grid ladder, to measure how many cores the baseline actually engages
        ladder = {}
        for cx, cy in ((2, 2), (4, 4), (8, 5), (11, 10)):
            try:
                g = ttnn.CoreGrid(y=cy, x=cx)
                s = timed(lambda: ttnn.deallocate(ttnn.linear(
                    x, w, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                    core_grid=g)), dev, warm=2, pipe=2, reps=5)
                ladder[f"{cx}x{cy}"] = round(s * 1e6, 2)
            except Exception as e:                                        # noqa: BLE001
                ladder[f"{cx}x{cy}"] = str(e)[:60]
        print(f"  {site:16s} core_grid ladder {ladder}", flush=True)
        out[site] = {"n_cols": ncol, "nt": nt, "legs": rows, "core_grid_ladder": ladder}
        ttnn.deallocate(w)
    ttnn.deallocate(x)
    res["sites"] = out


def arm_l1out(dev, ckc, res):
    """Candidate 1: can a pair-track projection put its 48.82 MB output in L1?"""
    free0 = l1_free_per_bank(dev)
    res["l1_free_empty_device"] = {"largest_contig_per_bank": free0[0], "total_free_per_bank": free0[1],
                                   "bank_bytes": _l1_bank_bytes(), "banks": _BANKS}
    print(f"  empty-device L1: contig {free0[0]/1024:.1f} kB/bank, free {free0[1]/1024:.1f} kB/bank,"
          f" banks={res['l1_free_empty_device']['banks']}", flush=True)
    x = _pair_x(dev)
    w = ttnn.from_torch(torch.randn(C_Z, C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=DRAM)
    m_tiles = TOK * (NPAD // 32)
    out_bytes = m_tiles * 32 * C_Z * 2
    res["l1out_output_bytes"] = out_bytes
    banks = res["l1_free_empty_device"]["banks"]
    print(f"  projection output = {out_bytes/1e6:.2f} MB = {out_bytes/banks/1024:.1f} kB/bank",
          flush=True)
    rows = []
    # A) the whole 48.82 MB output in L1, empty device
    for lbl, cfg in [("core_grid", None)] + [
            (f"1d_bw{bw}_obh{obh}", pair_cfg(m_tiles, 8, 8, bw, obh, out_l1=True))
            for bw in (8, 1) for obh in (5, 2, 1)]:
        if lbl != "core_grid" and cfg is None:
            rows.append({"scope": "empty_device_full_output", "leg": lbl, "err": "config refused by L1 budget"})
            continue
        try:
            kw = dict(memory_config=L1, dtype=ttnn.bfloat16, compute_kernel_config=ckc)
            fn = ((lambda: ttnn.linear(x, w, core_grid=CORE_GRID_MAIN, **kw)) if cfg is None
                  else (lambda c: lambda: ttnn.linear(x, w, program_config=c, **kw))(cfg))
            s = timed(fn, dev, warm=2, pipe=2, reps=5)
            rows.append({"scope": "empty_device_full_output", "leg": lbl, "us": round(s * 1e6, 2),
                         "tflops": round(2 * m_tiles * 32 * C_Z * C_Z / s / 1e12, 2)})
            print(f"  L1-out full output {lbl:16s} {s*1e6:9.2f} us", flush=True)
        except Exception as e:                                            # noqa: BLE001
            rows.append({"scope": "empty_device_full_output", "leg": lbl, "err": str(e)[:150]})
            print(f"  L1-out full output {lbl:16s} REFUSED {str(e)[:110]}", flush=True)
    # B) how much output CAN live in L1 at all, empty device: bisect on row count
    lo, hi, ok = 0, m_tiles, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0:
            break
        try:
            xs = ttnn.from_torch(torch.randn(mid * 32, C_Z), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            o = ttnn.linear(xs, w, memory_config=L1, dtype=ttnn.bfloat16,
                            compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
            ttnn.synchronize_device(dev)
            ttnn.deallocate(o)
            ttnn.deallocate(xs)
            ok, lo = mid, mid + 1
        except Exception:                                                 # noqa: BLE001
            hi = mid - 1
    res["l1out_max_rows_tiles"] = ok
    res["l1out_max_output_MB"] = round(ok * 32 * C_Z * 2 / 1e6, 2)
    print(f"  largest L1 output this card accepts on an EMPTY device: {ok} m-tiles = "
          f"{res['l1out_max_output_MB']} MB (the projection needs {out_bytes/1e6:.2f} MB)", flush=True)
    res["l1out"] = rows
    ttnn.deallocate(x)
    ttnn.deallocate(w)


def arm_fuse(dev, ckc, res):
    """Candidate 2: qkv (nt=24) + g (nt=8) + triangle_bias (nt=1) on the same normed x."""
    x = ttnn.from_torch(torch.randn(TOK, TOK, C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=DRAM)
    n_heads = 4
    wq = torch.randn(C_Z, 3 * C_Z)
    wg = torch.randn(C_Z, C_Z)
    wb = torch.randn(C_Z, n_heads)
    tq = ttnn.from_torch(wq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    tg = ttnn.from_torch(wg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    tb = ttnn.from_torch(wb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    # the fused weight pads the bias block to a whole tile so the split is tile-aligned
    wfuse = torch.zeros(C_Z, 3 * C_Z + C_Z + 32)
    wfuse[:, :3 * C_Z] = wq
    wfuse[:, 3 * C_Z:4 * C_Z] = wg
    wfuse[:, 4 * C_Z:4 * C_Z + n_heads] = wb
    tf = ttnn.from_torch(wfuse, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=DRAM)
    m_tiles = TOK * (NPAD // 32)
    mm = lambda a, b: ttnn.experimental.minimal_matmul(
        a, b, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc)

    def leg_sep():
        q = mm(x, tq); g = mm(x, tg); b = mm(x, tb)
        ttnn.deallocate(q); ttnn.deallocate(g); ttnn.deallocate(b)

    def leg_fused_only():
        y = mm(x, tf)
        ttnn.deallocate(y)

    def leg_fused_split():
        y = mm(x, tf)
        q = y[:, :, :3 * C_Z]
        g = y[:, :, 3 * C_Z:4 * C_Z]
        b = y[:, :, 4 * C_Z:]
        ttnn.deallocate(y); ttnn.deallocate(q); ttnn.deallocate(g); ttnn.deallocate(b)

    rows = {}
    for lbl, fn in (("separate_3", leg_sep), ("fused_only", leg_fused_only),
                    ("fused_plus_split", leg_fused_split)):
        try:
            s = timed(fn, dev, warm=2, pipe=2, reps=5)
            rows[lbl] = round(s * 1e6, 2)
            print(f"  fuse {lbl:20s} {s*1e6:9.2f} us", flush=True)
        except Exception as e:                                            # noqa: BLE001
            rows[lbl] = str(e)[:140]
            print(f"  fuse {lbl:20s} FAILED {str(e)[:110]}", flush=True)
    # per-op breakdown of the separate leg, so the class shares are on the record
    for lbl, w_, nt in (("qkv_nt24", tq, 24), ("g_nt8", tg, 8), ("bias_nt1", tb, 1)):
        s = timed(lambda: ttnn.deallocate(mm(x, w_)), dev, warm=2, pipe=2, reps=5)
        wr = m_tiles * 32 * nt * 32 * 2
        rows[lbl] = {"us": round(s * 1e6, 2), "write_GBs": round(wr / s / 1e9, 1),
                     "tflops": round(2 * m_tiles * 32 * C_Z * nt * 32 / s / 1e12, 2)}
        print(f"  fuse {lbl:20s} {s*1e6:9.2f} us  {wr/s/1e9:7.1f} GB/s written", flush=True)
    # parity of the fused arm against the three separate calls
    q1, g1, b1 = mm(x, tq), mm(x, tg), mm(x, tb)
    y = mm(x, tf)
    ttnn.synchronize_device(dev)
    par = {}
    for lbl, ref, sl in (("qkv", q1, slice(0, 3 * C_Z)), ("g", g1, slice(3 * C_Z, 4 * C_Z)),
                         ("bias", b1, slice(4 * C_Z, 4 * C_Z + n_heads))):
        a = ttnn.to_torch(ref)
        b = ttnn.to_torch(y)[:, :, sl]
        if lbl == "bias":
            a = a[:, :, :n_heads]
        par[lbl] = {"torch_equal": bool(torch.equal(a, b)),
                    "max_abs": float((a.to(torch.float32) - b.to(torch.float32)).abs().max())}
    print("  fuse parity vs separate:", json.dumps(par), flush=True)
    for t in (q1, g1, b1, y, x, tq, tg, tb, tf):
        ttnn.deallocate(t)
    res["fuse"] = {"legs": rows, "parity_vs_separate": par}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["wroof", "sites", "l1out", "fuse"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    dg = dev.compute_with_storage_grid_size()
    global _BANKS
    _BANKS = dg.x * dg.y
    res = {"arm": a.arm, "compute_grid": f"{dg.x}x{dg.y}",
           "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
           "l1_bank_bytes": _l1_bank_bytes()}
    {"wroof": arm_wroof, "sites": arm_sites, "l1out": arm_l1out, "fuse": arm_fuse}[a.arm](dev, ckc, res)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
