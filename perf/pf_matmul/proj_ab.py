#!/usr/bin/env python3
"""Pairformer matmul dataflow (L1): are the pair-track projections held at ~26% of their
write roof by in0_block_w=1, and does an explicit 1D program config move them on qb1's 13x10?

Every shape here is a real Pairformer class at 298 aa (N_tok padded to 320), taken from W1's
ledger table (M/K/N in tiles). Roofs are measured in this same process on this card.
"""
import argparse, json, math, os, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
T = 32


def med(xs):
    return sorted(xs)[len(xs) // 2]


def timed(dev, fn, warm=2, pipe=6, reps=5):
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
        o.append((time.perf_counter() - t0) * 1e3 / pipe)
    return med(o)


def roofs(dev, ckc):
    """compute (square, deep K), DRAM read, DRAM write -- measured here, not inherited."""
    out = {}
    n = 4096
    a = ttnn.from_torch(torch.randn(1, 1, n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    b = ttnn.from_torch(torch.randn(1, 1, n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    ms = timed(dev, lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
        a, b, memory_config=DRAM, compute_kernel_config=ckc)), pipe=4)
    out["compute_square_TFLOPs"] = round(2 * n ** 3 / 1e9 / ms, 2)
    ttnn.deallocate(a); ttnn.deallocate(b)
    # DRAM read: DRAM -> L1 clone is not possible at 64 MB, so use DRAM->DRAM and DRAM->L1 small
    nb = 32 * 1024 * 1024  # 32 MB tensor
    rows = nb // (2 * 1024)
    t = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    ms = timed(dev, lambda: ttnn.deallocate(ttnn.clone(t, memory_config=DRAM)), pipe=4)
    out["dram_rw_clone_GBs"] = round(2 * nb / 1e9 / (ms / 1e3), 1)
    out["dram_rw_clone_ms"] = round(ms, 4)
    ttnn.deallocate(t)
    return out


# (label, M, K, N, production backend, calls/block, site)
def shapes(cz):
    S = 320          # padded token dim at 298 aa
    M = S * S        # 102400 pair rows
    if cz == 256:    # protenix-v2
        return [
            ("trimul.out_proj p_out+g_out", M, cz, cz, "linear_cg", 2, "_trimul_out_proj"),
            ("triatt.out x_out",            M, cz, cz, "linear_cg", 2, "gate_and_project"),
            ("triatt.qkv",                  M, cz, 3 * 8 * 32, "minimal_matmul", 2, "TriAtt qkv"),
            ("triatt.gate g",               M, cz, cz, "minimal_matmul", 2, "TriAtt g"),
            ("triatt.triangle_bias",        M, cz, 32, "linear_cg", 2, "bias_weight"),
            ("trimul.in_proj gp_fused",     M, cz, 128, "minimal_matmul", 8, "gp_in_fused"),
            ("transition.down",             32 * S, 1024, cz, "linear_cg", 10, "Transition x_dram"),
        ]
    return [  # opendde, c_z=384
        ("trimul.out_proj p_out+g_out", M, cz, cz, "linear_cg", 2, "_trimul_out_proj"),
        ("triatt.out x_out",            M, cz, cz, "linear_cg", 2, "gate_and_project"),
        ("triatt.qkv",                  M, cz, 3 * 12 * 32, "minimal_matmul", 2, "TriAtt qkv"),
        ("transition.up",               16 * S, cz, 4 * cz, "linear_cg", 20, "Transition x_1"),
        ("transition.down",             16 * S, 4 * cz, cz, "linear_cg", 20, "Transition x_dram"),
    ]


def divisors(n, cap):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cz", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    dev = get_device()
    ag = dev.compute_with_storage_grid_size()
    gx, gy = int(ag.x), int(ag.y)
    nc = gx * gy
    grid = ttnn.CoreGrid(y=gy, x=gx)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    print(f"grid {gx}x{gy} = {nc} cores", flush=True)
    R = roofs(dev, ckc)
    print("ROOFS", json.dumps(R), flush=True)

    torch.manual_seed(0)
    recs = []
    for label, M, K, N, prod, npb, site in shapes(a.cz):
        if a.only and a.only not in label:
            continue
        mt, kt, nt = M // T, K // T, math.ceil(N / T)
        at = torch.randn(1, 1, M, K) * 0.1
        bt = torch.randn(1, 1, K, N) * 0.1
        ref = (at.float() @ bt.float())
        ta = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        tb = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        gflop = 2 * M * K * N / 1e9
        rd = (M * K + K * N) * 2 / 1e9
        wr = M * N * 2 / 1e9
        ai = gflop * 1e9 / ((rd + wr) * 1e9)
        pcm = next((p for p in range(-(-mt // nc), mt + 1) if mt % p == 0), mt)
        print(f"\n=== {label}  [{M}x{K}]@[{K}x{N}]  mt={mt} kt={kt} nt={nt} pcm={pcm} "
              f"blocks={mt//pcm}/{nc} cores  prod={prod}  {npb}/block  AI={ai:.1f} FLOP/byte ===", flush=True)
        rec = {"label": label, "M": M, "K": K, "N": N, "mt": mt, "kt": kt, "nt": nt,
               "per_core_M": pcm, "prod": prod, "calls_per_block": npb, "site": site,
               "AI": round(ai, 1), "gflop": gflop, "read_GB": rd, "write_GB": wr, "v": {}}

        def report(name, ms, y):
            rm = float((y - ref).pow(2).mean().sqrt())
            rec["v"][name] = {"ms": round(ms, 4), "TFLOPs": round(gflop / ms, 2),
                              "read_GBs": round(rd / (ms / 1e3), 1), "write_GBs": round(wr / (ms / 1e3), 1),
                              "rmsd": rm}
            print(f"  {name:26s} {ms:8.4f} ms  {gflop/ms:6.2f} TF/s  rd {rd/(ms/1e3):6.1f} "
                  f"wr {wr/(ms/1e3):6.1f} GB/s  rmsd {rm:.5f}", flush=True)
            return rec["v"][name]

        y = ttnn.linear(ta, tb, compute_kernel_config=ckc, core_grid=grid, memory_config=DRAM)
        y_cg = ttnn.to_torch(y).float(); ttnn.deallocate(y)
        ms_cg = timed(dev, lambda: ttnn.deallocate(
            ttnn.linear(ta, tb, compute_kernel_config=ckc, core_grid=grid, memory_config=DRAM)))
        report("linear_cg", ms_cg, y_cg)

        y = ttnn.experimental.minimal_matmul(ta, tb, memory_config=DRAM, compute_kernel_config=ckc)
        y_mm = ttnn.to_torch(y).float(); ttnn.deallocate(y)
        ms_mm = timed(dev, lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
            ta, tb, memory_config=DRAM, compute_kernel_config=ckc)))
        report("minimal_matmul", ms_mm, y_mm)

        y_prod = y_cg if prod == "linear_cg" else y_mm
        ms_prod = ms_cg if prod == "linear_cg" else ms_mm
        rec["ms_prod"] = ms_prod

        bws = sorted({1, 2, 4, min(8, kt), kt} & set(divisors(kt, kt)))
        obhs = divisors(pcm, 32)
        for bw in bws:
            for obh in obhs:
                for obw in sorted({nt, max(1, nt // 2), min(nt, 8), min(nt, 4)} & set(divisors(nt, nt))):
                    sh = max((h for h in range(min(4, obh), 0, -1) if obh % h == 0), default=1)
                    sw = max((w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0), default=1)
                    name = f"1d_bw{bw}_obh{obh}_obw{obw}"
                    if name in rec["v"]:
                        continue
                    try:
                        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
                            out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
                            per_core_M=pcm, per_core_N=nt, fuse_batch=True,
                            fused_activation=None, mcast_in0=False)
                        z = ttnn.linear(ta, tb, compute_kernel_config=ckc, memory_config=DRAM, program_config=pc)
                        zt = ttnn.to_torch(z).float(); ttnn.deallocate(z)
                        ms = timed(dev, lambda: ttnn.deallocate(ttnn.linear(
                            ta, tb, compute_kernel_config=ckc, memory_config=DRAM, program_config=pc)))
                    except Exception as e:
                        print(f"  {name:26s} SKIP {type(e).__name__}", flush=True)
                        continue
                    d = report(name, ms, zt)
                    d["eq_prod"] = bool(torch.equal(zt, y_prod))
                    d["x_vs_prod"] = round(ms_prod / ms, 3)
                    print(f"      -> {ms_prod/ms:.3f}x vs prod({prod})   eq_prod {d['eq_prod']}", flush=True)
        if rec["v"]:
            cand = {k: v for k, v in rec["v"].items() if k.startswith("1d_")}
            if cand:
                bk = min(cand, key=lambda k: cand[k]["ms"])
                rec["best"] = {"variant": bk, **cand[bk]}
                saved = (ms_prod - cand[bk]["ms"]) * npb * 480
                rec["ms_per_fold_saved"] = round(saved, 1)
                print(f"  BEST {bk}  {cand[bk]['ms']:.4f} ms  {cand[bk]['x_vs_prod']}x vs prod  "
                      f"-> {saved:.0f} ms/fold ({npb}/block x 480)", flush=True)
        recs.append(rec)
        ttnn.deallocate(ta); ttnn.deallocate(tb)

    p = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"proj_ab_cz{a.cz}.json")
    json.dump({"roofs": R, "grid": [gx, gy], "shapes": recs}, open(p, "w"), indent=1)
    print("\nwrote", p, flush=True)
    tot = sum(r.get("ms_per_fold_saved", 0) for r in recs)
    print(f"TOTAL ms/fold from program-config alone (c_z={a.cz}): {tot:.0f}", flush=True)


if __name__ == "__main__":
    main()
