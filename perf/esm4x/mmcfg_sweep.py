#!/usr/bin/env python3
"""EXEC step 1+2: the two config sweeps the plan screened, at ESMFold2's production shapes.

Three things, all off-fold, all `torch.equal` against the op they would replace:

  A. the trimul in-projection `minimal_matmul` at [1,512,512,256] x [256,1024] (kt=8, nt=32),
     which `_MM_BLOCK` has no entry for. Only K_block == kt is swept: any other K splits the
     contraction and changes the accumulation order, which is a different (non-bit-exact) fork.
  B. the channel matmul's out_subblock at the PRODUCTION fused shape (batch 256, group=8).
     perf/bigswing/mmcfg/tri_matmul_subblock_qb2c0.json already swept it at batch 32 and the
     shipped 1x1 won; this re-runs it where the fold actually is so the NO-GO is measured here.
  C. the gated move's roof: the same reader on a contiguous [1,512,512,256] source vs the fused
     [1,512,512,1024] one. If the contiguous move also sits near 180 GB/s the limit is the SFPU
     the kernel carries; if it reaches the back move's 268.6 GB/s the limit is the strided read
     of two slices out of a 1024-channel tensor, and a reader change is the fix.

Screen only. A per-call ratio here is not a fold gain; the fold A/B decides.
"""
import argparse, importlib.metadata as im, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RP

MB = 2 ** 20
WARM, REPS = 2, 5


def ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def divisors(n, cap=64):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def med(dev, fn, n=REPS, warm=WARM):
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    ts, keep = [], None
    for i in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if i < n - 1 and isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
        else:
            keep = o
    return st.median(ts) * 1e3, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--sizes", default="512,320,298")
    a = ap.parse_args()

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    CK = ckc()
    L, CZ, NW = a.L, 256, 1024
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "ttnn": im.version("ttnn"),
         "loadavg": open("/proc/loadavg").read().split()[:3], "warm": WARM, "reps": REPS,
         "note": "screen; per-call ratios, not fold gains", "A_inproj": {}, "B_subblock": {},
         "C_move": {}}
    torch.manual_seed(0)
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
    MC = ttnn.DRAM_MEMORY_CONFIG

    # ---------------- A. the in-projection --------------------------------------------------
    x = f(torch.randn(1, L, L, CZ, dtype=torch.bfloat16))
    w = f(torch.randn(CZ, NW, dtype=torch.bfloat16))
    mt, kt, nt = L * L // 32, CZ // 32, NW // 32
    base_ms, o = med(dev, lambda: ttnn.experimental.minimal_matmul(
        x, w, memory_config=MC, dtype=ttnn.bfloat16, compute_kernel_config=CK))
    ref = ttnn.to_torch(o); ttnn.deallocate(o)
    site = {"in": [1, L, L, CZ], "w": [CZ, NW], "m_tiles": mt, "k_tiles": kt, "n_tiles": nt,
            "base_ms": round(base_ms, 4), "arms": []}
    print(f"== A in-projection [1,{L},{L},{CZ}]x[{CZ},{NW}] mt={mt} kt={kt} nt={nt} "
          f"base {base_ms:.4f} ms", flush=True)
    cands = []
    for M in divisors(mt, 32):
        for N in divisors(nt):
            subs = {(1, 1)}
            for sh in divisors(M, 8):
                for sw in divisors(N, 8):
                    if sh * sw <= 8:
                        subs.add((sh, sw))
            best = max(subs, key=lambda s: s[0] * s[1])
            for s in ({best, (1, 1)} if best != (1, 1) else {(1, 1)}):
                cands.append((M, kt, N, s[0], s[1]))
    cands.sort()
    print(f"   {len(cands)} candidates (K_block == kt only)", flush=True)
    for M, K, N, sh, sw in cands:
        row = {"M": M, "K": K, "N": N, "sh": sh, "sw": sw}
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh, subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(*T.COMPUTE_GRID_MAIN))
        try:
            ms, o = med(dev, lambda: ttnn.experimental.minimal_matmul(
                x, w, memory_config=MC, dtype=ttnn.bfloat16, compute_kernel_config=CK, config=cfg))
            got = ttnn.to_torch(o); ttnn.deallocate(o)
            row.update(ms=round(ms, 4), speedup=round(base_ms / ms, 4),
                       exact=bool(torch.equal(got, ref)))
            del got
            print(f"   M={M:3d} K={K} N={N:2d} sub={sh}x{sw}  {ms:8.4f} ms  "
                  f"{row['speedup']:.4f}x exact={row['exact']}", flush=True)
        except Exception as e:                                            # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:110]}"
        site["arms"].append(row)
    ok = sorted([r for r in site["arms"] if r.get("exact")], key=lambda r: -r["speedup"])
    site["best"] = ok[:5]
    R["A_inproj"]["L%d" % L] = site
    del ref
    ttnn.deallocate(x); ttnn.deallocate(w)
    if ok:
        print(f"   BEST bit-exact: {ok[0]}", flush=True)

    # the winner's bit-exactness at the other sizes the fold runs (_MM_BLOCK is shared)
    if ok:
        M, K, N, sh, sw = ok[0]["M"], ok[0]["K"], ok[0]["N"], ok[0]["sh"], ok[0]["sw"]
        for S in [int(s) for s in a.sizes.split(",") if int(s) != L]:
            xs = f(torch.randn(1, S, S, CZ, dtype=torch.bfloat16))
            ws = f(torch.randn(CZ, NW, dtype=torch.bfloat16))
            mts = -(-S * S // 32)
            rec = {"m_tiles": mts, "declined_by_key": bool(mts % M)}
            try:
                b_ms, ob = med(dev, lambda: ttnn.experimental.minimal_matmul(
                    xs, ws, memory_config=MC, dtype=ttnn.bfloat16, compute_kernel_config=CK), n=3)
                rb = ttnn.to_torch(ob); ttnn.deallocate(ob)
                rec["base_ms"] = round(b_ms, 4)
                if not rec["declined_by_key"]:
                    cfg = ttnn.MinimalMatmulConfig(
                        M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh,
                        subblock_w=sw,
                        compute_with_storage_grid_size=ttnn.CoreCoord(*T.COMPUTE_GRID_MAIN))
                    c_ms, oc = med(dev, lambda: ttnn.experimental.minimal_matmul(
                        xs, ws, memory_config=MC, dtype=ttnn.bfloat16, compute_kernel_config=CK,
                        config=cfg), n=3)
                    rc = ttnn.to_torch(oc); ttnn.deallocate(oc)
                    rec.update(ms=round(c_ms, 4), speedup=round(b_ms / c_ms, 4),
                               exact=bool(torch.equal(rb, rc)))
                    del rc
                del rb
            except Exception as e:                                        # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:110]}"
            R["A_inproj"]["L%d" % S] = rec
            print(f"   size {S}: {rec}", flush=True)
            ttnn.deallocate(xs); ttnn.deallocate(ws)
    Path(a.out).write_text(json.dumps(R, indent=1))

    # ---------------- B. the channel matmul at the production fused shape --------------------
    pair_mb = L * L * CZ * 2 / MB
    aa = f(torch.randn(1, CZ, L, L, dtype=torch.bfloat16))
    bb = f(torch.randn(1, CZ, L, L, dtype=torch.bfloat16))
    slt = (L + 31) // 32
    pc0 = T._triangle_mul_program_config(slt)
    print(f"== B channel matmul [1,{CZ},{L},{L}] shipped pc: in0_block_w={pc0.in0_block_w} "
          f"per_core=({pc0.per_core_M},{pc0.per_core_N}) sub=({pc0.out_subblock_h},"
          f"{pc0.out_subblock_w})", flush=True)

    def pc(sh, sw):
        gx, gy = T.COMPUTE_GRID_MAIN
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=pc0.in0_block_w,
            out_subblock_h=sh, out_subblock_w=sw, out_block_h=pc0.out_block_h,
            out_block_w=pc0.out_block_w, per_core_M=pc0.per_core_M, per_core_N=pc0.per_core_N,
            transpose_mcast=False, fused_activation=None, fuse_batch=False)

    ms0, o = med(dev, lambda: ttnn.matmul(aa, bb, compute_kernel_config=CK, memory_config=MC,
                                          program_config=pc0, dtype=ttnn.bfloat16))
    refm = ttnn.to_torch(o); ttnn.deallocate(o)
    R["B_subblock"]["shipped_1x1"] = {"ms": round(ms0, 4), "speedup": 1.0, "exact": True,
                                      "GBps": round(3 * pair_mb * MB / (ms0 * 1e-3) / 1e9, 1)}
    print(f"   shipped_1x1 {ms0:8.4f} ms", flush=True)
    for sh in (1, 2):
        for sw in (1, 2):
            if (sh, sw) == (1, 1):
                continue
            key = f"sub{sh}x{sw}"
            try:
                ms, o = med(dev, lambda: ttnn.matmul(aa, bb, compute_kernel_config=CK,
                                                     memory_config=MC, program_config=pc(sh, sw),
                                                     dtype=ttnn.bfloat16))
                got = ttnn.to_torch(o); ttnn.deallocate(o)
                R["B_subblock"][key] = {"ms": round(ms, 4), "speedup": round(ms0 / ms, 4),
                                        "exact": bool(torch.equal(got, refm))}
                del got
            except Exception as e:                                        # noqa: BLE001
                R["B_subblock"][key] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}
            print(f"   {key} {R['B_subblock'][key]}", flush=True)
    del refm
    ttnn.deallocate(aa); ttnn.deallocate(bb)
    Path(a.out).write_text(json.dumps(R, indent=1))

    # ---------------- C. the gated move's roof ----------------------------------------------
    gp = f(torch.randn(1, L, L, 4 * CZ, dtype=torch.bfloat16))   # the fused projection
    sc = CZ
    assert RP.eligible_gated(gp, sc, MC), "E6 declines -- screen invalid"
    ms_g, o = med(dev, lambda: RP.reblock_permute_gated(gp, 2 * sc, 0, sc, memory_config=MC))
    ttnn.deallocate(o)
    R["C_move"]["gated_from_fused_1024"] = {
        "ms": round(ms_g, 4), "MB": round(3 * pair_mb, 1),
        "GBps": round(3 * pair_mb * MB / (ms_g * 1e-3) / 1e9, 1)}
    print(f"== C gated move (fused 1024 src) {ms_g:8.4f} ms  "
          f"{R['C_move']['gated_from_fused_1024']['GBps']} GB/s", flush=True)
    narrow = f(torch.randn(1, L, L, CZ, dtype=torch.bfloat16))
    for label, fn, mb in (
            ("plain_fwd_from_256", lambda: RP.reblock_permute(narrow, memory_config=MC),
             2 * pair_mb),
            ("back_from_256", lambda: RP.reblock_permute_back(
                f(torch.randn(1, CZ, L, L, dtype=torch.bfloat16)), memory_config=MC),
             2 * pair_mb)):
        try:
            if not (RP.eligible(narrow, MC) if "plain" in label else True):
                R["C_move"][label] = {"ineligible": True}
                print(f"   {label}: INELIGIBLE", flush=True)
                continue
            ms, o = med(dev, fn, n=3)
            ttnn.deallocate(o)
            R["C_move"][label] = {"ms": round(ms, 4), "MB": round(mb, 1),
                                  "GBps": round(mb * MB / (ms * 1e-3) / 1e9, 1)}
        except Exception as e:                                            # noqa: BLE001
            R["C_move"][label] = {"error": f"{type(e).__name__}: {str(e)[:140]}"}
        print(f"   {label}: {R['C_move'][label]}", flush=True)
    R["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(R, indent=1))
    print("\nwrote " + str(a.out), flush=True)


if __name__ == "__main__":
    main()
