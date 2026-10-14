#!/usr/bin/env python3
"""OpenFold3 bf16 DiT linears at 512 tokens: the shipped call vs minimal_matmul configs, HiFi4 and HiFi2.

The WH census of the 512-token --fast fold (whcen, .114, 1000 MHz) puts the DiT's linears at 2.55 s
of 16.4 s device time, every one HiFi4 with fp32 accumulation through `ops.linear` -> `ttnn.linear`
with `core_grid=CORE_GRID_MAIN` and no program config:

    768 x 3072   107 us/call  (2.42 GFLOP, ~23 TF/s)
    384 x  768   19200 calls 0.78 s (~40 us each)
    768 x 1536   0.49 s   1536 x 768 0.21 s   768 x 1024 0.17 s   1024 x 768 0.16 s

23 TF/s is well under a 72-core Wormhole's HiFi4 matmul rate, so the program config, not the
math, is the first suspect; lever dit_hifi2 bought only 0.16-0.35 s of the expected ~1.2 s, which
says the same. Each site: the shipped call (control), the shipped call at HiFi2, then
minimal_matmul over (M, K, N) blocks at HiFi4 and, for the best few, HiFi2 and HiFi2 with bf16
dest (8-tile subblocks). `exact` is torch.equal against the shipped HiFi4 output; every arm also
carries rel L2 against a float64 reference.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402

T = 512
SITES = [  # name, K, N, bias
    ("ff_in", 768, 3072, False),
    ("ada_s", 384, 768, True),
    ("k768_n1536", 768, 1536, False),
    ("k1536_n768", 1536, 768, False),
    ("k768_n1024", 768, 1024, False),
    ("k1024_n768", 1024, 768, False),
]
WARM, REPS = 2, 5
M_VALS = (1, 2, 4, 8, 16)
K_VALS = (1, 2, 4, 8)
N_VALS = (1, 2, 4, 8)


def subblocks(M, N, dst):
    opts = [(h, w) for h in range(1, 9) for w in range(1, 9)
            if M % h == 0 and N % w == 0 and h * w <= dst]
    best = max(h * w for h, w in opts)
    return [o for o in opts if o[0] * o[1] == best][:2]


def timed(dev, fn, inner):
    for _ in range(WARM):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(inner):
            o = fn(); ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / inner)
    o = fn(); ttnn.synchronize_device(dev)
    return statistics.median(ts), o


def rel_l2(got, ref64):
    d = got.double() - ref64
    return float(d.norm() / ref64.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--sites", default=",".join(s[0] for s in SITES))
    a = ap.parse_args()
    dev = tt.get_device()
    gx, gy = tt.COMPUTE_GRID_MAIN
    wh = dev.arch() == ttnn.Arch.WORMHOLE_B0
    cls = ttnn.types.WormholeComputeKernelConfig if wh else ttnn.types.BlackholeComputeKernelConfig
    ckc = lambda fid, acc: cls(math_fidelity=fid, math_approx_mode=False,  # noqa: E731
                               fp32_dest_acc_en=acc, packer_l1_acc=True)
    H4, H2 = ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2
    res = {"arch": str(dev.arch()), "grid": [gx, gy], "loadavg": open("/proc/loadavg").read().split()[:3],
           "warm": WARM, "reps": REPS, "inner": a.inner, "tokens": T, "sites": []}
    print(f"arch {res['arch']} grid {gx}x{gy}", flush=True)
    torch.manual_seed(0)
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,  # noqa: E731
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    want = set(a.sites.split(","))
    for name, K, N, has_bias in SITES:
        if name not in want:
            continue
        xt = torch.randn(1, T, K, dtype=torch.bfloat16)
        wt = (torch.randn(K, N) / K ** 0.5).to(torch.bfloat16)
        bt = torch.randn(1, N, dtype=torch.bfloat16) if has_bias else None
        ref64 = xt.double()[0] @ wt.double() + (bt.double() if bt is not None else 0)
        x, w = up(xt), up(wt)
        b = up(bt) if bt is not None else None
        flop = 2 * T * K * N

        def shipped(c):
            return lambda: ttnn.linear(x, w, bias=b, compute_kernel_config=c, core_grid=tt.CORE_GRID_MAIN)

        def mm(cfg, c):
            return lambda: ttnn.experimental.minimal_matmul(
                x, w, bias_tensor=b, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                compute_kernel_config=c, config=cfg)

        base_s, o = timed(dev, shipped(ckc(H4, True)), a.inner)
        ref = ttnn.to_torch(o).reshape(T, N); ttnn.deallocate(o)
        site = {"site": name, "K": K, "N": N, "bias": has_bias, "base_us": 1e6 * base_s,
                "base_tflops": flop / base_s / 1e12, "base_rel_l2_f64": rel_l2(ref, ref64), "arms": []}
        print(f"\n== {name} [{T}x{K}]x[{K}x{N}] shipped {1e6*base_s:.1f} us "
              f"({site['base_tflops']:.1f} TF/s) relL2 {site['base_rel_l2_f64']:.2e}", flush=True)

        def arm(tag, fn, **kv):
            row = {"arm": tag, **kv}
            try:
                s, o = timed(dev, fn, a.inner)
                got = ttnn.to_torch(o).reshape(T, N); ttnn.deallocate(o)
                row.update(us=1e6 * s, speedup=base_s / s, tflops=flop / s / 1e12,
                           exact=bool(torch.equal(got, ref)), rel_l2_f64=rel_l2(got, ref64))
                print(f"   {tag:10s} {kv}  {row['us']:.1f} us  {row['speedup']:.3f}x  "
                      f"exact={row['exact']} relL2 {row['rel_l2_f64']:.2e}", flush=True)
            except Exception as e:                                          # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            site["arms"].append(row)
            return row

        arm("ship_h2", shipped(ckc(H2, True)))
        arm("ship_h2_b16", shipped(ckc(H2, False)))
        mt, kt, nt = T // 32, K // 32, N // 32
        cands = [(Mb, Kb, Nb, sh, sw) for Mb in M_VALS for Kb in K_VALS for Nb in N_VALS
                 if Mb <= mt and Kb <= kt and Nb <= nt and kt % Kb == 0
                 for sh, sw in subblocks(Mb, Nb, 4)]
        mcfg = lambda Mb, Kb, Nb, sh, sw: ttnn.MinimalMatmulConfig(  # noqa: E731
            M_block_size=Mb, K_block_size=Kb, N_block_size=Nb, subblock_h=sh, subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
        arm("mm_h4", mm(None, ckc(H4, True)), M=0)
        for Mb, Kb, Nb, sh, sw in cands:
            arm("mm_h4", mm(mcfg(Mb, Kb, Nb, sh, sw), ckc(H4, True)), M=Mb, K=Kb, N=Nb, sh=sh, sw=sw)
        top = sorted([r for r in site["arms"] if r["arm"] == "mm_h4" and "us" in r and r.get("M")],
                     key=lambda r: r["us"])[:4]
        for r in top:
            Mb, Kb, Nb = r["M"], r["K"], r["N"]
            arm("mm_h2", mm(mcfg(Mb, Kb, Nb, r["sh"], r["sw"]), ckc(H2, True)), M=Mb, K=Kb, N=Nb)
            for sh, sw in subblocks(Mb, Nb, 8):
                arm("mm_h2_b16", mm(mcfg(Mb, Kb, Nb, sh, sw), ckc(H2, False)), M=Mb, K=Kb, N=Nb, sh=sh, sw=sw)
        ok = [r for r in site["arms"] if "us" in r]
        site["best"] = {t: min((r for r in ok if r["arm"] == t), key=lambda r: r["us"], default=None)
                        for t in ("ship_h2", "ship_h2_b16", "mm_h4", "mm_h2", "mm_h2_b16")}
        site["best_exact"] = min((r for r in ok if r["exact"]), key=lambda r: r["us"], default=None)
        print(f"   best: { {t: (r and round(r['us'], 1)) for t, r in site['best'].items()} }", flush=True)
        for t in (x, w) + ((b,) if b is not None else ()):
            ttnn.deallocate(t)
        res["sites"].append(site)
        Path(a.out).write_text(json.dumps(res, indent=1))
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
