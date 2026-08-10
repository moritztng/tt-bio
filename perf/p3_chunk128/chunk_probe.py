#!/usr/bin/env python3
"""p3-chunk128 probes: roofs on THIS card, the SDPA chunk ladder 64..320, the op-level numerical
screen, the grid ladder at the surviving rung, and the alignment penalty per rung.

Shapes are the ones a live 298 aa protenix-v2 fold issues: q/k/v [298, 8, 320, 32] padded from
logical 298, bias [1, 8, 320, 320] in DRAM. Device synchronised on BOTH sides of every timed
region. Results stream to --out so a timeout leaves finished stages on disk.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-chunk128 \
      python3 perf/p3_chunk128/chunk_probe.py --out perf/p3_chunk128/probe_c0.json
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
M, N, CZ, NH, HD = 298, 320, 256, 8, 32
CALLS = 1048          # 2 x 524, counted in a live fold by X1 and re-counted in my own fold runs
RUNGS = (64, 96, 128, 160, 256, 320)
RES = {}
DEV = None
CKC = None


def save(path):
    json.dump(RES, open(path, "w"), indent=1)


def timed(fn, warm=3, pipe=4, reps=7):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(DEV)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def us(x):
    return round(x * 1e6, 1)


def msfold(u, calls=CALLS):
    return round(u * calls / 1000.0, 1)


def T(shape, mc=DRAM, dt=ttnn.bfloat16, gen=None):
    t = torch.randn(*shape, generator=gen) if gen is not None else torch.randn(*shape)
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=dt, memory_config=mc)


def cfg(chunk, grid=(11, 10)):
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=False,
                                  q_chunk_size=chunk, k_chunk_size=chunk)


def stage(name, fn, out):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.perf_counter()
    try:
        RES[name] = fn()
    except Exception as e:                                             # noqa: BLE001
        RES[name] = {"error": f"{type(e).__name__}: {e}"[:600]}
        print("  ERR", RES[name]["error"], flush=True)
    print(f"  [{time.perf_counter() - t0:.1f}s]", flush=True)
    save(out)


# ---------------------------------------------------------------------------- roofs, on this card
def roofs():
    r = {"card": f"qb1 TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')}",
         "note": "measured on this card this pass, not inherited",
         "loadavg_at_start": os.getloadavg()}
    n = 4096
    a, b = T((1, 1, n, n)), T((1, 1, n, n))
    s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
                                                  memory_config=DRAM)), warm=2, pipe=3, reps=5)
    r["compute_square_4096_TFLOPs"] = round(2 * n ** 3 / s / 1e12, 2)
    ttnn.deallocate(a); ttnn.deallocate(b)
    print(f"  square 4096: {r['compute_square_4096_TFLOPs']} TFLOP/s", flush=True)

    rows = []
    for mb in (32, 64, 128):
        nrow = int(mb * 1e6 / 2) // 4096
        nb = nrow * 4096 * 2
        row = {"MB": round(nb / 1e6, 2)}
        xd = T((nrow, 4096), DRAM)
        row["read_GBs"] = round(nb / timed(lambda: ttnn.deallocate(
            ttnn.clone(xd, memory_config=L1)), warm=2, pipe=3, reps=5) / 1e9, 1)
        row["copy_rw_GBs"] = round(2 * nb / timed(lambda: ttnn.deallocate(
            ttnn.clone(xd, memory_config=DRAM)), warm=2, pipe=3, reps=5) / 1e9, 1)
        ttnn.deallocate(xd)
        try:
            xl = T((nrow, 4096), L1)
            row["write_GBs"] = round(nb / timed(lambda: ttnn.deallocate(
                ttnn.clone(xl, memory_config=DRAM)), warm=2, pipe=3, reps=5) / 1e9, 1)
            row["l1_to_l1_GBs"] = round(2 * nb / timed(lambda: ttnn.deallocate(
                ttnn.clone(xl, memory_config=L1)), warm=2, pipe=3, reps=5) / 1e9, 1)
            ttnn.deallocate(xl)
        except Exception as e:                                         # noqa: BLE001
            row["l1_err"] = str(e)[:80]
        rows.append(row)
        print("  " + json.dumps(row), flush=True)
    r["dram"] = rows
    r["read_peak_GBs"] = max(x.get("read_GBs", 0) for x in rows)
    r["copy_peak_GBs"] = max(x.get("copy_rw_GBs", 0) for x in rows)
    r["clone_write_peak_GBs"] = max(x.get("write_GBs", 0) for x in rows)
    r["l1_to_l1_peak_GBs"] = max(x.get("l1_to_l1_GBs", 0) for x in rows)
    best = 0.0
    for (m, k, nn) in ((M * N, 32, 2048), (M * N, 32, 4096)):
        try:
            a2, b2 = T((m, k), L1), T((k, nn), L1)
            s2 = timed(lambda: ttnn.deallocate(ttnn.matmul(a2, b2, compute_kernel_config=CKC,
                                                           memory_config=DRAM)),
                       warm=2, pipe=3, reps=5)
            gbs = round(m * nn * 2 / s2 / 1e9, 1)
            r[f"mm_writer_{m}x{k}x{nn}_GBs"] = gbs
            best = max(best, gbs)
            ttnn.deallocate(a2); ttnn.deallocate(b2)
        except Exception as e:                                         # noqa: BLE001
            r[f"mm_writer_{m}x{k}x{nn}_err"] = str(e)[:120]
    r["mm_writer_write_peak_GBs"] = best
    r["machine_balance_FLOP_per_byte"] = round(
        r["compute_square_4096_TFLOPs"] * 1e12 / (r["read_peak_GBs"] * 1e9), 1)
    print(f"  ROOFS read {r['read_peak_GBs']} copy {r['copy_peak_GBs']} l1l1 "
          f"{r['l1_to_l1_peak_GBs']} mm-write {best} GB/s "
          f"balance {r['machine_balance_FLOP_per_byte']} FLOP/B", flush=True)
    return r


# ----------------------------------------------------------------- the chunk ladder, 64 .. 320
def chunk_ladder():
    r = {"shape": "q/k/v [298, 8, 320, 32], bias [1, 8, 320, 320] DRAM, grid 11x10",
         "calls_per_fold": CALLS, "loadavg": os.getloadavg()}
    q, k, v = (T((M, NH, N, HD)) for _ in range(3))
    bias = T((1, NH, N, N), DRAM)

    def run(mask, c):
        return timed(lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, scale=HD ** -0.5,
            program_config=cfg(c))), warm=2, pipe=3, reps=5)

    for c in RUNGS:
        try:
            for lbl, mask in (("bias", bias), ("nobias", None)):
                r[f"chunk{c}_{lbl}_us"] = us(run(mask, c))
            r[f"chunk{c}_biasleg_us"] = round(r[f"chunk{c}_bias_us"] - r[f"chunk{c}_nobias_us"], 1)
            r[f"chunk{c}_biasleg_GBs"] = round(
                M * NH * N * N * 2 / (r[f"chunk{c}_biasleg_us"] * 1e-6) / 1e9, 1)
            r[f"chunk{c}_legal"] = True
            print(f"  chunk{c}: bias {r[f'chunk{c}_bias_us']} us  nobias "
                  f"{r[f'chunk{c}_nobias_us']} us  biasleg {r[f'chunk{c}_biasleg_us']} us",
                  flush=True)
        except Exception as e:                                         # noqa: BLE001
            r[f"chunk{c}_legal"] = False
            r[f"chunk{c}_error"] = f"{type(e).__name__}: {e}"[:400]
            print(f"  chunk{c}: ILLEGAL {r[f'chunk{c}_error']}", flush=True)
    base = r.get("chunk64_bias_us")
    for c in RUNGS:
        if r.get(f"chunk{c}_legal") and base:
            r[f"chunk{c}_speedup_vs64"] = round(base / r[f"chunk{c}_bias_us"], 3)
            r[f"chunk{c}_delta_us_per_call"] = round(base - r[f"chunk{c}_bias_us"], 1)
            r[f"chunk{c}_probe_ms_per_fold"] = msfold(base - r[f"chunk{c}_bias_us"])
            print(f"  chunk{c}: {r[f'chunk{c}_speedup_vs64']}x -> "
                  f"{r[f'chunk{c}_probe_ms_per_fold']} ms/fold (probe)", flush=True)
    for t in (q, k, v, bias):
        ttnn.deallocate(t)
    return r


# ------------------------------------- the op-level numerical SCREEN (NOT a fold figure)
def screen():
    """Same inputs, every chunk, compared against chunk 64's own output. Synthetic q/k/v, so this
    is a SCREEN. P1 measured 2.30 % at the block for chunk 320 and the fold said 2.962 A."""
    r = {"what": "SDPA output deviation vs chunk 64, same synthetic inputs, fold's own shape",
         "caveat": "screen only -- a block/op figure is not a fold figure"}
    g = torch.Generator().manual_seed(0)
    qt, kt, vt = (torch.randn(M, NH, N, HD, generator=g) for _ in range(3))
    bt = torch.randn(1, NH, N, N, generator=g)
    q, k, v = (ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16,
                               memory_config=DRAM) for x in (qt, kt, vt))
    bias = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16,
                           memory_config=DRAM)
    outs = {}
    for c in RUNGS:
        try:
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5,
                program_config=cfg(c))
            outs[c] = ttnn.to_torch(o).float()
            ttnn.deallocate(o)
        except Exception as e:                                         # noqa: BLE001
            r[f"chunk{c}_error"] = str(e)[:300]
    ref = outs.get(64)
    if ref is not None:
        x = ref.flatten().double()
        rms_ref = float(torch.sqrt((x ** 2).mean()))
        r["rms_of_chunk64_output"] = rms_ref
        for c, o in outs.items():
            y = o.flatten().double()
            d = y - x
            xc, yc = x - x.mean(), y - y.mean()
            r[f"chunk{c}"] = dict(
                rmsd=float(torch.sqrt((d ** 2).mean())),
                relative_rmsd=float(torch.sqrt((d ** 2).mean())) / rms_ref,
                max_abs_deviation=float(d.abs().max()),
                pcc=float((xc * yc).sum() / (xc.norm() * yc.norm())),
                torch_equal=bool(torch.equal(x, y)))
            print(f"  chunk{c}: rel RMSD {r[f'chunk{c}']['relative_rmsd']:.6f}  "
                  f"PCC {r[f'chunk{c}']['pcc']:.8f}  torch.equal "
                  f"{r[f'chunk{c}']['torch_equal']}", flush=True)
    for t in (q, k, v, bias):
        ttnn.deallocate(t)
    return r


# ------------------------------------------------------------- grid ladder -> core-equivalents
def grid_ladder(chunks=(64, 128)):
    r = {}
    q, k, v = (T((M, NH, N, HD)) for _ in range(3))
    bias = T((1, NH, N, N), DRAM)
    for c in chunks:
        lad = {}
        for g in ((1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (11, 10)):
            try:
                lad[f"{g[0]}x{g[1]}"] = us(timed(
                    lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5,
                        program_config=cfg(c, g))), warm=1, pipe=2, reps=3))
            except Exception as e:                                     # noqa: BLE001
                lad[f"{g[0]}x{g[1]}"] = str(e)[:80]
        r[f"grid_ladder_chunk{c}_us"] = lad
        if isinstance(lad.get("1x1"), float) and isinstance(lad.get("11x10"), float):
            r[f"core_equivalents_chunk{c}_of_110"] = round(lad["1x1"] / lad["11x10"], 1)
        print(f"  chunk{c} core-equivalents: {r.get(f'core_equivalents_chunk{c}_of_110')}",
              flush=True)
    for t in (q, k, v, bias):
        ttnn.deallocate(t)
    return r


# --------------------------------------------------- the alignment penalty, per surviving rung
def alignment(chunks=(64, 128)):
    r = {"method": "logical key length 298 vs 320, both inside a 320-padded buffer, same config"}
    for c in chunks:
        for klog in (298, 320):
            q = T((M, NH, N, HD)); kk = T((M, NH, klog, HD)); vv = T((M, NH, klog, HD))
            bias = T((1, NH, N, klog), DRAM)
            try:
                r[f"chunk{c}_klogical{klog}_us"] = us(timed(
                    lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
                        q, kk, vv, attn_mask=bias, is_causal=False, scale=HD ** -0.5,
                        program_config=cfg(c))), warm=2, pipe=3, reps=5))
            except Exception as e:                                     # noqa: BLE001
                r[f"chunk{c}_klogical{klog}_err"] = str(e)[:200]
            for t in (q, kk, vv, bias):
                ttnn.deallocate(t)
        a, b = r.get(f"chunk{c}_klogical298_us"), r.get(f"chunk{c}_klogical320_us")
        if a and b:
            r[f"chunk{c}_penalty_us"] = round(a - b, 2)
            r[f"chunk{c}_penalty_ms_per_fold"] = msfold(a - b)
            print(f"  chunk{c}: 298 {a} vs 320 {b} us -> "
                  f"{r[f'chunk{c}_penalty_ms_per_fold']} ms/fold", flush=True)
    return r


def main():
    global DEV, CKC
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    DEV = get_device()
    CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    RES["meta"] = dict(host=os.uname().nodename, visible=os.environ.get("TT_VISIBLE_DEVICES"),
                       ttnn=ttnn.__version__ if hasattr(ttnn, "__version__") else "unknown",
                       core_grid_main=str(CORE_GRID_MAIN), loadavg=os.getloadavg(),
                       rungs=list(RUNGS))
    only = set(x for x in args.only.split(",") if x)
    for name, fn in (("roofs", roofs), ("chunk_ladder", chunk_ladder), ("screen", screen),
                     ("grid_ladder", grid_ladder), ("alignment", alignment)):
        if not only or name in only:
            stage(name, fn, args.out)
    RES["meta"]["loadavg_end"] = os.getloadavg()
    save(args.out)
    print("\nwrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
