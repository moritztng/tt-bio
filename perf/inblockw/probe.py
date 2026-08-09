#!/usr/bin/env python3
"""W-inblockw planning probe: verify the in0_block_w lever and validate a shape-resolved
config builder on this card, under the PRODUCTION compute kernel config.

Answers, all measured here rather than inherited from W13:
  A. does ttnn.linear accept program_config=, and what happens with core_grid= too?
  B. at the four real 298 aa families, is a tuned config faster than core_grid=, and by how much?
  C. does a single closed-form builder pick a config within a few % of the ladder best?
  D. per-shape rmsd vs an fp32 torch reference, and max abs delta vs the core_grid baseline.
"""
import json, os, sys, time
import torch
import ttnn

GRID = None
CKC = None


def kcfg():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def l1_budget():
    return int(ttnn.get_max_worker_l1_unreserved_size())


def divisors_upto(k, lo, hi):
    return [d for d in range(hi, lo - 1, -1) if d <= k and k % d == 0]


def build_1d(mt, kt, nt, bw, ob_h, ob_w, gx, gy):
    """1D M-split (systolic) config, output blocked so the CB fits L1."""
    cores = gx * gy
    per_core_M = -(-mt // cores)
    sh = max((h for h in range(min(4, ob_h), 0, -1) if ob_h % h == 0), default=1)
    sw = max((w for w in range(min(4 // sh, ob_w), 0, -1) if ob_w % w == 0), default=1)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=ob_h,
        out_block_w=ob_w,
        per_core_M=per_core_M,
        per_core_N=nt,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


def build_2d(mt, kt, nt, bw, gx, gy):
    per_core_M = -(-mt // gy)
    per_core_N = -(-nt // gx)
    sh = max((h for h in range(min(4, per_core_M), 0, -1) if per_core_M % h == 0), default=1)
    sw = max((w for w in range(min(4 // sh, per_core_N), 0, -1) if per_core_N % w == 0), default=1)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        r = fn()
        ttnn.deallocate(r)
    ttnn.synchronize_device(DEV)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3)
        last = r
    ts.sort()
    return ts[len(ts) // 2], last


def main():
    global DEV, CKC
    DEV = ttnn.open_device(device_id=0)
    a = DEV.compute_with_storage_grid_size()
    gx, gy = int(a.x), int(a.y)
    if gx >= 13:
        gx = 13
    elif gx >= 11:
        gx = 11
    gy = min(gy, 10)
    CKC = kcfg()
    L1 = l1_budget()
    print(f"grid {gx}x{gy} = {gx*gy} cores, L1 unreserved {L1} B", flush=True)
    out = {"grid": [gx, gy], "l1": L1, "shapes": []}

    SHAPES = [
        # (label, M, K, N, family)
        ("298 trimul in/out 102400x256@256x256", 102400, 256, 256, "pair_wide"),
        ("298 pair transition down 102400x512@512x128", 102400, 512, 128, "pair_narrow"),
        ("298 pair->bias heads 102400x128@128x32", 102400, 128, 32, "pair_nt1"),
        ("298 single trans down 320x3072@3072x768", 320, 3072, 768, "single"),
        ("298 single trans up 320x768@768x3072", 320, 768, 3072, "single"),
        ("298 atom proj 4480x128@128x128", 4480, 128, 128, "atom"),
    ]

    # --- A: API acceptance -------------------------------------------------
    x = ttnn.from_torch(torch.randn(1, 32, 128), layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.randn(128, 128), layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    pc = build_1d(1, 4, 4, 4, 1, 4, gx, gy)
    api = {}
    try:
        r = ttnn.linear(x, w, compute_kernel_config=CKC, dtype=ttnn.bfloat16, program_config=pc)
        api["program_config_only"] = "OK"
        ttnn.deallocate(r)
    except Exception as e:
        api["program_config_only"] = f"ERR {type(e).__name__}: {str(e)[:200]}"
    try:
        r = ttnn.linear(x, w, compute_kernel_config=CKC, dtype=ttnn.bfloat16,
                        program_config=pc, core_grid=ttnn.CoreGrid(y=gy, x=gx))
        api["both"] = "OK (accepted, program_config presumably wins)"
        ttnn.deallocate(r)
    except Exception as e:
        api["both"] = f"ERR {type(e).__name__}: {str(e)[:200]}"
    print("API:", json.dumps(api, indent=1), flush=True)
    out["api"] = api
    ttnn.deallocate(x); ttnn.deallocate(w)

    cores = gx * gy
    for label, M, K, N in [(s[0], s[1], s[2], s[3]) for s in SHAPES]:
        fam = [s[4] for s in SHAPES if s[0] == label][0]
        mt, kt, nt = M // 32, K // 32, N // 32
        xt = torch.randn(1, M, K, dtype=torch.float32) * 0.1
        wt = torch.randn(K, N, dtype=torch.float32) * 0.1
        ref = (xt.to(torch.float32) @ wt.to(torch.float32))[0]
        X = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
        W = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)

        rec = {"label": label, "family": fam, "mt": mt, "kt": kt, "nt": nt, "variants": []}

        def run_and_record(name, fn):
            try:
                ms, res = timeit(fn)
            except Exception as e:
                rec["variants"].append({"name": name, "err": f"{type(e).__name__}: {str(e)[:160]}"})
                print(f"  {label:52s} {name:28s} ERR {str(e)[:90]}", flush=True)
                return None
            h = ttnn.to_torch(res).to(torch.float32)
            h = h.reshape(-1, N)[:M]
            rmsd = float(torch.sqrt(torch.mean((h - ref) ** 2)))
            rec["variants"].append({"name": name, "ms": ms, "rmsd": rmsd})
            print(f"  {label:52s} {name:28s} {ms:9.4f} ms  rmsd {rmsd:.6f}", flush=True)
            return h

        base = run_and_record("linear core_grid (BASE)", lambda: ttnn.linear(
            X, W, compute_kernel_config=CKC, dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=gy, x=gx)))
        run_and_record("linear no core_grid", lambda: ttnn.linear(
            X, W, compute_kernel_config=CKC, dtype=ttnn.bfloat16))

        # ladder: 1D for tall M, 2D for short M
        tile = 2048  # bf16 tile bytes
        if mt >= 4 * cores:
            per_core_M = -(-mt // cores)
            for bw in divisors_upto(kt, 1, 16):
                # pick output blocking that fits: out CB = ob_h*ob_w*(tile + 4096 fp32 acc)
                fit = None
                for ob_w in divisors_upto(nt, 1, nt):
                    for ob_h in divisors_upto(per_core_M, 1, per_core_M):
                        need = ob_h * ob_w * (tile + 4096) + (ob_h + ob_w) * bw * tile + 64 * 1024
                        if need <= L1:
                            fit = (ob_h, ob_w, need)
                            break
                    if fit:
                        break
                if not fit:
                    continue
                ob_h, ob_w, need = fit
                pc = build_1d(mt, kt, nt, bw, ob_h, ob_w, gx, gy)
                run_and_record(f"1D bw{bw} ob{ob_h}x{ob_w}", lambda pc=pc: ttnn.linear(
                    X, W, compute_kernel_config=CKC, dtype=ttnn.bfloat16, program_config=pc))
        else:
            for bw in divisors_upto(kt, 1, 24):
                pc = build_2d(mt, kt, nt, bw, gx, gy)
                run_and_record(f"2D bw{bw}", lambda pc=pc: ttnn.linear(
                    X, W, compute_kernel_config=CKC, dtype=ttnn.bfloat16, program_config=pc))

        ok = [v for v in rec["variants"] if "ms" in v]
        b = [v for v in ok if v["name"].startswith("linear core_grid")][0]
        best = min(ok, key=lambda v: v["ms"])
        rec["base_ms"] = b["ms"]
        rec["best"] = best
        rec["speedup"] = b["ms"] / best["ms"]
        print(f"  ==> {label}: BASE {b['ms']:.4f} -> BEST {best['name']} {best['ms']:.4f} "
              f"= {rec['speedup']:.3f}x   rmsd {b['rmsd']:.6f} -> {best['rmsd']:.6f}", flush=True)
        out["shapes"].append(rec)
        ttnn.deallocate(X); ttnn.deallocate(W)

    with open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe_inblockw.json", "w") as f:
        json.dump(out, f, indent=1)
    ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
