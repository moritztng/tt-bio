#!/usr/bin/env python3
"""MinimalMatmulConfig sweep at AF2-IG's two biggest `minimal_matmul` sites, on Wormhole (8x8).

Adapted from perf/bigswing/mmcfg_sweep.py (the BH sweep behind `_MM_BLOCK`). The AF2-IG census at
195 tokens (.114:~/spd/spd-gap-bind/out/af2igwh_census_62d89f627/s0-opsall/census.json) puts:

    _attend_pair    q/k/v + gate projections   832 calls  0.386 s/fold synced
    _in_proj_matmul trimul in-projection       832 calls  0.355 s/fold synced

Shapes as the code runs them at 195 tokens (normal mode, AF2's kernel config = HiFi4,
fp32_dest_acc_en, packer_l1_acc; bf16 everywhere):

    qkv   [195,195,128] x [128,384]            no bias  out bf16 DRAM  (`_attend_pair`, AF2 biased)
    gate  [195,195,128] x [128,128]            no bias  out bf16 DRAM  (g bias added after, `_O`)
    inprj [1,195,195,128] x [128,256] + [256]  bias     out bf16 DRAM  (`_in_proj_matmul`, bias_i)

mt = ceil(195*195/32) = 1189 = 29*41, so the BH table's `mt % M` guard rejects every M_block > 1 and
none of these sites gets a config today: the op's default (8,8,8,2,2) runs, which pads K from 4 to
8 tiles. The op itself does not need M_block | mt (it pads M per core), so M is swept freely here.

Each arm: warm 2, median of 5 reps, each rep `--inner` back-to-back calls between two
device syncs (per-call ms reported). `exact` is torch.equal against the unconfigured default's
output; every arm also carries rel L2 against a float64 torch reference.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402

S = 195
SITES = [
    ("qkv", (S, S, 128), (128, 384), False),
    ("gate", (S, S, 128), (128, 128), False),
    ("inproj", (1, S, S, 128), (128, 256), True),
]
WARM, REPS = 2, 5
M_VALS = (1, 2, 3, 4, 6, 8, 12, 16)
K_VALS = (1, 2, 4)
N_VALS = (1, 2)          # N per core is 1-2 tiles at every site on an 8-wide axis


def subblocks(M, N):
    """Largest-area legal subblocks (fp32 dest acc: <= 4 dst tiles), every tie."""
    opts = [(h, w) for h in range(1, 5) for w in range(1, 5)
            if M % h == 0 and N % w == 0 and h * w <= 4]
    best = max(h * w for h, w in opts)
    return [o for o in opts if o[0] * o[1] == best]


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
    return statistics.median(ts), sorted(ts), o


def rel_l2(got, ref64):
    d = got.double() - ref64
    return float(d.norm() / ref64.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--sites", default="qkv,gate,inproj")
    a = ap.parse_args()
    dev = tt.get_device()
    gx, gy = tt.COMPUTE_GRID_MAIN
    arch = str(dev.arch())
    ckc_cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
               else ttnn.types.BlackholeComputeKernelConfig)
    ckc = ckc_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                  fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"arch": arch, "grid": [gx, gy], "ttnn": getattr(ttnn, "__version__", "?"),
           "loadavg": open("/proc/loadavg").read().split()[:3], "warm": WARM, "reps": REPS,
           "inner": a.inner, "ckc": "HiFi4 fp32_dest_acc packer_l1_acc", "sites": []}
    print(f"arch {arch} grid {gx}x{gy}", flush=True)
    torch.manual_seed(0)
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,  # noqa: E731
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    want = set(a.sites.split(","))
    for name, ish, wsh, has_bias in SITES:
        if name not in want:
            continue
        rows = 1
        for d in ish[:-1]:
            rows *= d
        mt, kt, nt = (rows + 31) // 32, ish[-1] // 32, wsh[-1] // 32
        xt = torch.randn(*ish, dtype=torch.bfloat16)
        wt = (torch.randn(*wsh, dtype=torch.bfloat16) / 128 ** 0.5).to(torch.bfloat16)
        bt = torch.randn(wsh[-1], dtype=torch.bfloat16) if has_bias else None
        ref64 = xt.double().reshape(rows, -1) @ wt.double()
        if bt is not None:
            ref64 = ref64 + bt.double()
        ref64 = ref64.reshape(*ish[:-1], wsh[-1])
        x, w = up(xt), up(wt)
        b = up(bt) if bt is not None else None

        def run(cfg):
            return lambda: ttnn.experimental.minimal_matmul(
                x, w, bias_tensor=b, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                compute_kernel_config=ckc, config=cfg)

        base_s, base_ts, o = timed(dev, run(None), a.inner)
        ref = ttnn.to_torch(o); ttnn.deallocate(o)
        site = {"site": name, "in": list(ish), "w": list(wsh), "bias": has_bias, "m_tiles": mt,
                "k_tiles": kt, "n_tiles": nt, "base_ms": 1e3 * base_s,
                "base_reps_ms": [1e3 * t for t in base_ts],
                "base_rel_l2_f64": rel_l2(ref, ref64), "arms": []}
        print(f"\n== {name} {ish}x{wsh} bias={has_bias} mt={mt} kt={kt} nt={nt}  "
              f"default {1e3*base_s:.4f} ms  relL2 {site['base_rel_l2_f64']:.3e}", flush=True)
        cands = [(8, 8, 8, 2, 2)]       # == the op's own default: a control, must be exact and ~1.0x
        for M in M_VALS:
            for K in K_VALS:
                for N in N_VALS:
                    if N > max(nt, 1):
                        continue
                    for sh, sw in subblocks(M, N):
                        cands.append((M, K, N, sh, sw))
        site["n_candidates"] = len(cands)
        print(f"   {len(cands)} candidates", flush=True)
        for M, K, N, sh, sw in cands:
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
            row = {"M": M, "K": K, "N": N, "sh": sh, "sw": sw}
            try:
                s, ts, o = timed(dev, run(cfg), a.inner)
                got = ttnn.to_torch(o); ttnn.deallocate(o)
                row.update(ms=1e3 * s, reps_ms=[1e3 * t for t in ts], speedup=base_s / s,
                           exact=bool(torch.equal(got, ref)),
                           max_abs_vs_default=float((got.float() - ref.float()).abs().max()),
                           rel_l2_f64=rel_l2(got, ref64))
                print(f"   M={M:2d} K={K} N={N} sub={sh}x{sw}  {row['ms']:.4f} ms  "
                      f"{row['speedup']:.4f}x  exact={row['exact']}  relL2 {row['rel_l2_f64']:.3e}",
                      flush=True)
            except Exception as e:                                          # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                print(f"   M={M:2d} K={K} N={N} sub={sh}x{sw}  ERROR {row['error'][:100]}", flush=True)
            site["arms"].append(row)
        ok = sorted([r for r in site["arms"] if r.get("exact")], key=lambda r: -r["speedup"])
        nx = sorted([r for r in site["arms"] if "ms" in r and not r["exact"]],
                    key=lambda r: -r["speedup"])
        site["best_exact"], site["best_nonexact"] = ok[:5], nx[:5]
        print(f"   best exact: {[(r['M'], r['K'], r['N'], r['sh'], r['sw'], round(r['ms'], 4), round(r['speedup'], 4)) for r in ok[:5]]}")
        print(f"   best non-exact: {[(r['M'], r['K'], r['N'], r['sh'], r['sw'], round(r['ms'], 4), round(r['speedup'], 4), '%.2e' % r['rel_l2_f64']) for r in nx[:3]]}",
              flush=True)
        for t in (x, w) + ((b,) if b is not None else ()):
            ttnn.deallocate(t)
        res["sites"].append(site)
        Path(a.out).write_text(json.dumps(res, indent=1))

    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
