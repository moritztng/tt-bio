#!/usr/bin/env python3
"""Validate the candidate `_tuned_matmul_config` against a full 1D+2D ladder.

For each shape: BASE (`linear(core_grid=)`), no-core_grid, the helper's choice, and every legal
config on both ladders. Reports helper-vs-BASE, helper-vs-ladder-best, and rmsd vs an fp32
torch reference for all three.
"""
import json, sys, time
import torch
import ttnn
from tuned_cfg import _tuned_matmul_config, _subblock, _largest_divisor

DEV = None
CKC = None


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(DEV)
    ts, last = [], None
    for _ in range(n):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3)
        last = r
    ts.sort()
    return ts[len(ts) // 2], last


def ladder_1d(mt, kt, nt, gx, gy, l1, tile):
    cores = gx * gy
    per_core_M = -(-mt // cores)
    out = []
    for bw in [d for d in range(min(32, kt), 0, -1) if kt % d == 0]:
        for ob_w in [d for d in range(nt, 0, -1) if nt % d == 0][:3]:
            for ob_h in [d for d in range(per_core_M, 0, -1) if per_core_M % d == 0][:4]:
                sh, sw = _subblock(ob_h, ob_w)
                out.append((f"1D bw{bw} ob{ob_h}x{ob_w}",
                            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                                compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
                                out_subblock_h=sh, out_subblock_w=sw,
                                out_block_h=ob_h, out_block_w=ob_w,
                                per_core_M=per_core_M, per_core_N=nt,
                                fuse_batch=True, fused_activation=None, mcast_in0=False)))
    return out


def ladder_2d(mt, kt, nt, gx, gy, l1, tile):
    per_core_M = -(-mt // gy)
    per_core_N = -(-nt // gx)
    out = []
    for bw in [d for d in range(min(32, kt), 0, -1) if kt % d == 0]:
        sh, sw = _subblock(per_core_M, per_core_N)
        out.append((f"2D bw{bw}", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
            out_subblock_h=sh, out_subblock_w=sw,
            out_block_h=per_core_M, out_block_w=per_core_N,
            per_core_M=per_core_M, per_core_N=per_core_N,
            transpose_mcast=False, fused_activation=None, fuse_batch=False)))
    return out


SHAPES = [
    # The shapes the 298 aa Pairformer Transition ACTUALLY runs, read off perf/inblockw/census.py.
    # W13 and validate.py measured the unchunked mt=3200 pair shape; the real fold chunks the
    # transition, so M is 300 (or 280) tiles, which the first helper declined.
    ("trans up  mt300 9600x256@256x1024", 9600, 256, 1024),
    ("trans down mt300 9600x1024@1024x256", 9600, 1024, 256),
    ("trans up  mt280 8960x256@256x1024", 8960, 256, 1024),
    ("trans down mt280 8960x1024@1024x256", 8960, 1024, 256),
    ("trans up  mt300 nt16 9600x256@256x512", 9600, 256, 512),
    ("trans down mt300 kt16 9600x512@512x256", 9600, 512, 256),
    # opendde: c_z = 384, so kt = 12 and the transition is 4x wider
    ("opendde trans up mt300 9600x384@384x1536", 9600, 384, 1536),
    ("opendde trans down mt300 9600x1536@1536x384", 9600, 1536, 384),
]


def main():
    global DEV, CKC
    DEV = ttnn.open_device(device_id=0)
    a = DEV.compute_with_storage_grid_size()
    gx, gy = int(a.x), int(a.y)
    gx = 13 if gx >= 13 else (11 if gx >= 11 else gx)
    gy = min(gy, 10)
    CKC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    L1 = int(ttnn.get_max_worker_l1_unreserved_size())
    tile = 2048
    print(f"grid {gx}x{gy}={gx*gy} cores, L1 {L1}", flush=True)
    res = {"grid": [gx, gy], "l1": L1, "shapes": []}

    for label, M, K, N in SHAPES:
        mt, kt, nt = M // 32, K // 32, N // 32
        xt = torch.randn(1, M, K) * 0.1
        wt = torch.randn(K, N) * 0.1
        ref = (xt.float() @ wt.float())[0]
        X = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
        W = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
        rec = {"label": label, "mt": mt, "kt": kt, "nt": nt, "variants": []}

        def go(name, fn):
            try:
                ms, r = timeit(fn)
            except Exception as e:
                rec["variants"].append({"name": name, "err": str(e)[:110]})
                return
            h = ttnn.to_torch(r).float().reshape(-1, N)[:M]
            rec["variants"].append({"name": name, "ms": ms,
                                    "rmsd": float(torch.sqrt(torch.mean((h - ref) ** 2)))})
            ttnn.deallocate(r)

        go("BASE core_grid", lambda: ttnn.linear(X, W, compute_kernel_config=CKC,
                                                 dtype=ttnn.bfloat16,
                                                 core_grid=ttnn.CoreGrid(y=gy, x=gx)))
        go("no core_grid", lambda: ttnn.linear(X, W, compute_kernel_config=CKC,
                                              dtype=ttnn.bfloat16))
        hp = _tuned_matmul_config(mt, kt, nt, 2, (gx, gy), L1)
        rec["helper_cfg"] = (None if hp is None else
                             {"kind": type(hp).__name__.replace("Matmul", "").replace("ProgramConfig", ""),
                              "in0_block_w": hp.in0_block_w,
                              "out_block_h": getattr(hp, "out_block_h", None),
                              "out_block_w": getattr(hp, "out_block_w", None),
                              "per_core_M": hp.per_core_M, "per_core_N": hp.per_core_N,
                              "sub": [hp.out_subblock_h, hp.out_subblock_w]})
        if hp is not None:
            go("HELPER", lambda: ttnn.linear(X, W, compute_kernel_config=CKC,
                                             dtype=ttnn.bfloat16, program_config=hp))
        for nm, pc in ladder_1d(mt, kt, nt, gx, gy, L1, tile) + ladder_2d(mt, kt, nt, gx, gy, L1, tile):
            go(nm, lambda pc=pc: ttnn.linear(X, W, compute_kernel_config=CKC,
                                             dtype=ttnn.bfloat16, program_config=pc))

        ok = [v for v in rec["variants"] if "ms" in v]
        base = next(v for v in ok if v["name"] == "BASE core_grid")
        helper = next((v for v in ok if v["name"] == "HELPER"), None)
        best = min(ok, key=lambda v: v["ms"])
        rec["base_ms"] = base["ms"]
        rec["best_name"], rec["best_ms"] = best["name"], best["ms"]
        rec["helper_ms"] = helper["ms"] if helper else None
        rec["helper_vs_base"] = base["ms"] / helper["ms"] if helper else None
        rec["helper_vs_best"] = helper["ms"] / best["ms"] if helper else None
        rec["rmsd_base"], rec["rmsd_helper"] = base["rmsd"], helper["rmsd"] if helper else None
        print(f"{label:38s} mt{mt:5d} kt{kt:3d} nt{nt:3d} | BASE {base['ms']:8.4f} "
              f"HELPER {(helper['ms'] if helper else float('nan')):8.4f} "
              f"({(rec['helper_vs_base'] or 0):5.3f}x) | BEST {best['name']:16s} {best['ms']:8.4f} "
              f"(helper/best {(rec['helper_vs_best'] or 0):5.3f}) | rmsd {base['rmsd']:.6f} -> "
              f"{(helper['rmsd'] if helper else float('nan')):.6f} | cfg {rec['helper_cfg']}", flush=True)
        res["shapes"].append(rec)
        ttnn.deallocate(X); ttnn.deallocate(W)

    with open(sys.argv[1], "w") as f:
        json.dump(res, f, indent=1)
    ttnn.close_device(DEV)


main()
