#!/usr/bin/env python3
"""Which regime is the head-major qkv projection in: per-byte or per-output-tile?

The plumbing that lets `tt_bio/triatt_qkv.py` and `tt_bio/mm_dualnoc.py` take a block-float
operand already exists (`mm_generic.fast_dtypes_ok`, `TT_BIO_TRIATT_B8`, default off). Whether it
PAYS on this projection has never been measured, and the two figures the campaign owns disagree by
3x: `c14-bfp8-fastpath` measured 0.93-0.98 of the byte cut realized on Blackhole streaming ops,
`b2z-bfp8-narrow` measured 0.26-0.30 on Wormhole matmul-class sites. This is a matmul on
Blackhole, so it inherits neither.

`c13-matmul-rate` fitted the fold's thin-K matmul keys as t(kt) = a + b*kt with a = 73.9 % of the
time at kt = 4, measured with DRAM traffic removed. The open question is NOT whether that intercept
exists -- it does -- but whether it is byte-proportional. The output write and the in0 block read
are both per-output-tile and both scale with the storage width, so a nonzero intercept does not by
itself mean a width cut buys nothing. So: fit the same ladder at two widths and compare intercepts.

  --part ladder   K-ladder at the shipped block config, bf16 vs bfp8 destination vs bfp8 all.
                  a_b8/a_b16 ~ 0.53 => the per-tile cost is byte-bound and bfp8 pays at kt=4.
                  a_b8/a_b16 ~ 1.00 => the intercept is genuinely fixed and bfp8 can only cut the
                  b*kt term, which is 26 % of the time at kt=4.
  --part prod     the production key through `triatt_qkv.qkv_heads` and `qkvgb_heads` themselves,
                  plus the stock minimal_matmul + nlp_create_qkv_heads control, so KERNEL-PATH
                  reports the program that ran and not an assumption about it.
  --part acc      float64 reference. The hard stop is a transform that is WRONG rather than
                  imprecise, so the bfp8 arms are scored against float64 and never against bf16.

Arms are interleaved rep by rep with the order reversed on odd reps, the bf16 arm runs twice as its
own A/A twin, and the clock is forced and sampled DURING.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock"))

import clk  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=("ladder", "prod", "prodclean", "scatter", "acc"), required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--mhz", type=int, default=1350)
ap.add_argument("--n", type=int, default=512, help="pair side, so M = n*n rows")
ap.add_argument("--cz", type=int, default=128)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--head-dim", type=int, default=32)
ap.add_argument("--ladder-mtiles", type=int, default=2048)
ap.add_argument("--ladder-kt", default="4,8,16,32")
ap.add_argument("--acc-m", type=int, default=64, help="pair side for the float64 arm")
A = ap.parse_args()
OUT = Path(A.out) if A.out else HERE / ("%s.json" % A.part)

MGD = ensure_p300_mesh_descriptor()
dev = ttnn.open_device(device_id=0)

from tt_bio import mm_dualnoc as D8                      # noqa: E402
from tt_bio import mm_generic as G                       # noqa: E402
from tt_bio import triatt_qkv as Q                       # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, _mm_block_for  # noqa: E402

B16, B8 = ttnn.bfloat16, ttnn.bfloat8_b
WIDTH = {B16: 2.0, B8: 1.0625}                # bfp8_b tile is 1088 B over 1024 datums
DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = 32
GRID = tuple(COMPUTE_GRID_MAIN)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
CK = G.ckc_args(CKC)

res = {"part": A.part, "grid": list(GRID), "cores": GRID[0] * GRID[1], "mgd": MGD,
       "mhz_requested": A.mhz, "reps": A.reps, "n": A.n, "cz": A.cz,
       "heads": A.heads, "head_dim": A.head_dim, "arms": {}, "errors": {}}


def t(shape, dt, scale=1.0):
    return ttnn.from_torch(torch.randn(*shape, dtype=torch.float32) * scale,
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt, memory_config=DRAM)


def timeit(fn):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) * 1e3


def run_arms(arms):
    """`arms` is a list of (name, callable, nbytes). Interleaved, order reversed on odd reps."""
    live = []
    for name, fn, nb in arms:
        try:
            fn()
            ttnn.synchronize_device(dev)
            live.append((name, fn, nb))
        except Exception as e:                                            # noqa: BLE001
            res["errors"][name] = "%s: %s" % (type(e).__name__, str(e)[:400])
            print("REFUSED %-18s %s" % (name, str(e)[:160]), flush=True)
    res["capable"] = [n for n, _, _ in live]
    samples = {n: [] for n, _, _ in live}
    for rep in range(A.reps):
        for name, fn, _ in (live if rep % 2 == 0 else list(reversed(live))):
            samples[name].append(timeit(fn))
        print("rep %d" % rep, flush=True)
    for name, _, nb in live:
        v = samples[name]
        med = statistics.median(v)
        res["arms"][name] = {"ms_med": med, "ms_min": min(v), "ms": v, "bytes": nb,
                             "gbs": (nb / (med * 1e-3) / 1e9) if nb else None}
    return live


def fit(points):
    """Least squares t = a + b*kt over {kt: ms}. Returns (a, b, r2)."""
    xs = sorted(points)
    ys = [points[k] for k in xs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else None)


def force_clock():
    held = clk.force(A.mhz, clk.nodes_open_by_this_process())
    res["nodes_forced"] = held
    for _ in range(200):
        if all(clk.aiclk(nd) >= A.mhz - 5 for nd in held):
            break
        time.sleep(0.05)
    s = clk.Sampler(held[0])
    time.sleep(0.3)
    return held, s


def finish():
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in (v or {}).items() if kk != "ms"}
                      for k, v in res["arms"].items()}, indent=2))
    for k in ("aa_floor_pct", "fits", "ratios", "kernel_path", "clock", "errors"):
        if k in res:
            print(k, json.dumps(res[k]) if isinstance(res[k], dict) else res[k])
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------- ladder
if A.part == "ladder":
    # The shipped boltz2 entry for the qkv+gate weight, (kt, nt) = (4, 16): K_block = 4 is the
    # whole contraction there. Holding K_block at 4 while kt grows keeps the per-block schedule
    # the arm actually ships with and makes K_blocks = kt/4, so what the ladder varies is the
    # number of K blocks streamed and nothing else. Changing K_block per rung instead would move
    # the kernel config between rungs, which is the mistake `roof-cube-control-kernel-config-
    # must-match-arm` names (same cube 1.40x apart on config alone).
    BLK = (4, 4, 1, 4, 1)
    KTS = [int(x) for x in A.ladder_kt.split(",")]
    MT, NT = A.ladder_mtiles, 16
    M, N = MT * TILE, NT * TILE
    res.update({"block_cfg": list(BLK), "m_tiles": MT, "n_tiles": NT, "kts": KTS})

    # one operand set per (kt, dtype) so no arm pays an allocation inside the timed region
    T = {}
    for kt in KTS:
        K = kt * TILE
        for dt in (B16, B8):
            T[("x", kt, dt)] = t((M, K), dt)
            T[("w", kt, dt)] = t((K, N), dt, scale=0.05)
        for dt in (B16, B8):
            T[("o", kt, dt)] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([M, N]), dt, ttnn.TILE_LAYOUT, dev, DRAM)

    def mm(kt, xdt, wdt, odt):
        x, w, o = T[("x", kt, xdt)], T[("w", kt, wdt)], T[("o", kt, odt)]
        cfg = (BLK, GRID)

        def f():
            G.generic_minimal_matmul(dev, x, w, o, cfg, CK)
        nb = (WIDTH[xdt] * M * kt * TILE + WIDTH[wdt] * kt * TILE * N + WIDTH[odt] * M * N)
        return f, nb

    ARMS = []
    for kt in KTS:
        ARMS.append(("b16_kt%d" % kt,) + mm(kt, B16, B16, B16))
        ARMS.append(("b8dest_kt%d" % kt,) + mm(kt, B16, B16, B8))
        ARMS.append(("b8all_kt%d" % kt,) + mm(kt, B8, B8, B8))
    # the A/A twin: the same bf16 arm a second time, at the production rung
    ARMS.append(("b16_aa_kt%d" % KTS[0],) + mm(KTS[0], B16, B16, B16))

    held, sampler = force_clock()
    run_arms(ARMS)
    res["clock"] = sampler.stop()
    res["aiclk_after"] = {nd: clk.aiclk(nd) for nd in held}

    base = res["arms"].get("b16_kt%d" % KTS[0], {}).get("ms_med")
    aa = res["arms"].get("b16_aa_kt%d" % KTS[0], {}).get("ms_med")
    if base and aa:
        res["aa_floor_pct"] = abs(aa - base) / base * 100
    res["fits"] = {}
    for fam in ("b16", "b8dest", "b8all"):
        pts = {kt: res["arms"]["%s_kt%d" % (fam, kt)]["ms_med"] for kt in KTS
               if "%s_kt%d" % (fam, kt) in res["arms"]}
        if len(pts) >= 3:
            a, b, r2 = fit(pts)
            res["fits"][fam] = {"intercept_ms": a, "slope_ms_per_kt": b, "r2": r2,
                                "intercept_share_at_kt%d" % KTS[0]: a / pts[KTS[0]]}
    res["ratios"] = {}
    f16 = res["fits"].get("b16")
    for fam in ("b8dest", "b8all"):
        f8 = res["fits"].get(fam)
        if f16 and f8:
            res["ratios"][fam] = {
                "intercept_vs_b16": f8["intercept_ms"] / f16["intercept_ms"],
                "slope_vs_b16": f8["slope_ms_per_kt"] / f16["slope_ms_per_kt"],
            }
        nm = "%s_kt%d" % (fam, KTS[0]), "b16_kt%d" % KTS[0]
        if nm[0] in res["arms"] and nm[1] in res["arms"]:
            r = res["arms"][nm[0]]["ms_med"] / res["arms"][nm[1]]["ms_med"]
            br = res["arms"][nm[0]]["bytes"] / res["arms"][nm[1]]["bytes"]
            res["ratios"].setdefault(fam, {}).update({
                "time_vs_b16_at_production_kt": r, "byte_ratio": br,
                "realization": (1 - r) / (1 - br) if br != 1 else None})
    finish()

# ---------------------------------------------------------------------------- prod
elif A.part == "prod":
    # The production key: the pair tensor is [n, n, c_z] and the projection is a single
    # minimal_matmul over M = n*n rows. Two weights, both with a shipped _MM_BLOCK entry:
    #   qkv      [c_z, 3*heads*head_dim]  -> (kt, nt) = (4, 12)   triatt_qkv.qkv_heads
    #   qkv+g+b  [c_z, 3*h*d + h*d + 32]  -> (kt, nt) = (4, 17)   triatt_qkv.qkvgb_heads
    S, C, H, D = A.n, A.cz, A.heads, A.head_dim
    W_QKV = 3 * H * D
    # the trimul in-projection, `mm_dualnoc.in_proj`'s production key and the second site this
    # row owns: [512,512,256] x [256,1024], (kt, nt) = (8, 32) on _MM_DEFAULT
    CT, NT_T = 256, 1024
    res.update({"w_qkv": [C, W_QKV], "w_inproj": [CT, NT_T]})

    X = {dt: t((S, S, C), dt) for dt in (B16, B8)}
    Wq = {dt: t((C, W_QKV), dt, scale=0.05) for dt in (B16, B8)}
    XT = {dt: t((S, S, CT), dt) for dt in (B16, B8)}
    WT = {dt: t((CT, NT_T), dt, scale=0.05) for dt in (B16, B8)}
    res["block_qkv"] = list(_mm_block_for(Wq[B16]) or ())

    MM_CFG_SENTINEL = object()   # qkv_heads only checks `is not None`; the descriptor it reads
    #                              is _mm_block_for(w), not this.

    def bytes_for(xdt, wdt, odt, k, nt_out, nt_w):
        return (WIDTH[xdt] * S * S * k + WIDTH[wdt] * k * nt_w * TILE
                + WIDTH[odt] * S * S * nt_out * TILE)

    def qkv_arm(xdt, wdt, odt):
        def f():
            o = Q.qkv_heads(X[xdt], Wq[wdt], CKC, H, D, odt, MM_CFG_SENTINEL)
            if o is None:
                raise RuntimeError("qkv_heads declined: %r" % (dict(Q.REJECTS),))
            res.setdefault("out_dtypes", {})["qkv_%s" % odt] = str(o[0].dtype)
            if o[0].dtype != odt:
                raise RuntimeError("qkv_heads returned %s for a %s request" % (o[0].dtype, odt))
            for x in o:
                ttnn.deallocate(x)
        return f, bytes_for(xdt, wdt, odt, C, 3 * H * D // TILE, W_QKV // TILE)

    def inproj_arm(xdt, wdt, odt):
        def f():
            o = D8.in_proj(XT[xdt], WT[wdt], CKC, odt, DRAM)
            if o is None:
                raise RuntimeError("in_proj declined: %r" % (dict(D8.REJECTS),))
            # the gate takes `dtype` now, so the allocation has to honour it: a bf16 tensor
            # handed back for a bfp8 request is the silent half-conversion this row fixed
            res.setdefault("out_dtypes", {})["inproj_%s" % odt] = str(o.dtype)
            if o.dtype != odt:
                raise RuntimeError("in_proj returned %s for a %s request" % (o.dtype, odt))
            ttnn.deallocate(o)
        return f, bytes_for(xdt, wdt, odt, CT, NT_T // TILE, NT_T // TILE)

    def stock_arm(xdt, odt):
        """What runs when the head-major gate declines: minimal_matmul then the head split."""
        def f():
            y = ttnn.experimental.minimal_matmul(
                input_tensor=X[xdt], weight_tensor=Wq[B16], compute_kernel_config=CKC,
                dtype=odt, config=None)
            y = ttnn.unsqueeze(y, 1)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                y, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                memory_config=y.memory_config())
            ttnn.deallocate(y)
            for x in (q, k, v):
                ttnn.deallocate(x)
        return f, bytes_for(xdt, B16, odt, C, 3 * H * D // TILE, W_QKV // TILE) \
            + 2 * WIDTH[odt] * S * S * 3 * H * D    # the split reads and rewrites the whole thing

    ARMS = [
        ("qkv_b16",) + qkv_arm(B16, B16, B16),
        ("qkv_b16_aa",) + qkv_arm(B16, B16, B16),
        ("qkv_b8dest",) + qkv_arm(B16, B16, B8),
        ("qkv_b8act",) + qkv_arm(B8, B16, B8),
        ("qkv_b8all",) + qkv_arm(B8, B8, B8),
        ("inproj_b16",) + inproj_arm(B16, B16, B16),
        ("inproj_b8dest",) + inproj_arm(B16, B16, B8),
        ("inproj_b8all",) + inproj_arm(B8, B8, B8),
        ("stock_b16",) + stock_arm(B16, B16),
        ("stock_b8dest",) + stock_arm(B16, B8),
    ]

    served0 = list(Q.STATS)
    dserved0 = list(D8.STATS)
    held, sampler = force_clock()
    run_arms(ARMS)
    res["clock"] = sampler.stop()
    res["aiclk_after"] = {nd: clk.aiclk(nd) for nd in held}
    res["kernel_path"] = {
        "triatt_qkv_served_total": Q.STATS[0] - served0[0],
        "triatt_qkv_declined_total": Q.STATS[1] - served0[1],
        "triatt_qkv_rejects": {str(k): v for k, v in Q.REJECTS.items()},
        "dualnoc_served_total": D8.STATS[0] - dserved0[0],
        "dualnoc_declined_total": D8.STATS[1] - dserved0[1],
        "dualnoc_rejects": {str(k): v for k, v in D8.REJECTS.items()},
        "program_served": ("tt_bio/kernels/triatt dm_in0_sender.cpp + dm_in1_sender_out.cpp "
                           "through ttnn.generic_op, block %s, grid %s"
                           % (res["block_qkv"], list(GRID))),
        "program_declined": ("ttnn.experimental.minimal_matmul (config=None -> _MM_DEFAULT "
                             "8,8,8,2,2) then ttnn.experimental.nlp_create_qkv_heads"),
    }
    base = res["arms"].get("qkv_b16", {}).get("ms_med")
    aa = res["arms"].get("qkv_b16_aa", {}).get("ms_med")
    if base and aa:
        res["aa_floor_pct"] = abs(aa - base) / base * 100
    res["ratios"] = {}
    for fam, ref in (("qkv_b8dest", "qkv_b16"), ("qkv_b8act", "qkv_b16"),
                     ("qkv_b8all", "qkv_b16"), ("inproj_b8dest", "inproj_b16"),
                     ("inproj_b8all", "inproj_b16"), ("stock_b8dest", "stock_b16"),
                     ("stock_b16", "qkv_b16")):
        if fam in res["arms"] and ref in res["arms"]:
            r = res["arms"][fam]["ms_med"] / res["arms"][ref]["ms_med"]
            br = res["arms"][fam]["bytes"] / res["arms"][ref]["bytes"]
            res["ratios"][fam + "_over_" + ref] = {
                "time": r, "byte_ratio": br,
                "realization": (1 - r) / (1 - br) if abs(br - 1) > 1e-9 else None}
    finish()

# ---------------------------------------------------------------------------- prodclean
elif A.part == "prodclean":
    # `--part prod` measures `qkv_heads` as production calls it, which includes allocating three
    # DRAM destinations per call. That cost is dtype-invariant, so it DILUTES the byte ratio, and
    # `prod` cannot say by how much. This part splits it: the same program at the same production
    # key with the destinations preallocated outside the timed region, plus an allocate-and-free
    # arm on its own so the invariant term is measured and not inferred.
    S, C, H, D = A.n, A.cz, A.heads, A.head_dim
    W_QKV = 3 * H * D
    X = {dt: t((S, S, C), dt) for dt in (B16, B8)}
    Wq = {dt: t((C, W_QKV), dt, scale=0.05) for dt in (B16, B8)}
    blk = _mm_block_for(Wq[B16])
    cfg = (blk, GRID)
    pad = [int(d) for d in X[B16].padded_shape]
    DEFS = {"HEAD_MAJOR_MT": pad[-2] // TILE}
    res.update({"w_qkv": [C, W_QKV], "block_qkv": list(blk or ()), "defines": DEFS})
    OUTS = {dt: [ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, H, S, D]), dt, ttnn.TILE_LAYOUT, dev, DRAM) for _ in range(3)]
        for dt in (B16, B8)}

    def nbytes(xdt, wdt, odt):
        return (WIDTH[xdt] * S * S * C + WIDTH[wdt] * C * W_QKV
                + WIDTH[odt] * S * S * W_QKV)

    def pre(xdt, wdt, odt):
        def f():
            G.generic_minimal_matmul(dev, X[xdt], Wq[wdt], OUTS[odt], cfg, CK, DEFS,
                                     Q.KERNEL_DIR)
        return f, nbytes(xdt, wdt, odt)

    def alloc_only(odt):
        def f():
            o = [ttnn.allocate_tensor_on_device(
                ttnn.Shape([S, H, S, D]), odt, ttnn.TILE_LAYOUT, dev, DRAM) for _ in range(3)]
            for y in o:
                ttnn.deallocate(y)
        return f, 0

    ARMS = [
        ("pre_b16",) + pre(B16, B16, B16),
        ("pre_b16_aa",) + pre(B16, B16, B16),
        ("pre_b8dest",) + pre(B16, B16, B8),
        ("pre_b8all",) + pre(B8, B8, B8),
        ("alloc_b16",) + alloc_only(B16),
        ("alloc_b8",) + alloc_only(B8),
    ]
    held, sampler = force_clock()
    run_arms(ARMS)
    res["clock"] = sampler.stop()
    res["aiclk_after"] = {nd: clk.aiclk(nd) for nd in held}
    base = res["arms"].get("pre_b16", {}).get("ms_med")
    aa = res["arms"].get("pre_b16_aa", {}).get("ms_med")
    if base and aa:
        res["aa_floor_pct"] = abs(aa - base) / base * 100
    res["ratios"] = {}
    for fam in ("pre_b8dest", "pre_b8all"):
        if fam in res["arms"] and base:
            r = res["arms"][fam]["ms_med"] / base
            br = res["arms"][fam]["bytes"] / res["arms"]["pre_b16"]["bytes"]
            res["ratios"][fam] = {"time": r, "byte_ratio": br,
                                  "realization": (1 - r) / (1 - br)}
    res["kernel_path"] = {
        "program": ("tt_bio/kernels/triatt dm_in0_sender.cpp + dm_in1_sender_out.cpp through "
                    "ttnn.generic_op, block %s, grid %s, defines %s, three DRAM destinations"
                    % (res["block_qkv"], list(GRID), DEFS)),
        "note": "same descriptor qkv_heads builds; only the destination allocation moved out",
    }
    finish()

# ---------------------------------------------------------------------------- scatter
elif A.part == "scatter":
    # Why this part exists. `--part prodclean` measured the production key with the destinations
    # preallocated and still returned 0.9509x on 0.6486x the bytes, while `--part ladder` returned
    # 0.7298x at the same kt with a destination preallocated the same way. So the invariant term is
    # NOT the per-call allocation (measured at 7.02 % of the call, `alloc_b16`) and not a generic
    # per-output-tile cost either. The two keys differ in TWO things at once: their m/n geometry,
    # and whether the write is one contiguous destination or three head-major ones.
    #
    # This part holds the geometry fixed and varies only the write. Both arms take the same in0,
    # the same in1, the same block entry, the same grid, the same number of output ELEMENTS and a
    # destination preallocated outside the timed region. `plain_*` writes one [S*S, W_QKV] tensor;
    # `hm_*` writes the three [S, H, S, D] head-major tensors the shipped projection writes. If the
    # head-major write is what refuses to narrow, plain narrows and hm does not.
    S, C, H, D = A.n, A.cz, A.heads, A.head_dim
    W_QKV = 3 * H * D
    X3 = {dt: t((S, S, C), dt) for dt in (B16, B8)}
    X2 = {dt: t((S * S, C), dt) for dt in (B16, B8)}
    Wq = {dt: t((C, W_QKV), dt, scale=0.05) for dt in (B16, B8)}
    blk = _mm_block_for(Wq[B16])
    cfg = (blk, GRID)
    DEFS = {"HEAD_MAJOR_MT": int(X3[B16].padded_shape[-2]) // TILE}
    res.update({"w_qkv": [C, W_QKV], "block_qkv": list(blk or ()), "defines": DEFS,
                "out_elems": S * S * W_QKV})
    HM = {dt: [ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, H, S, D]), dt, ttnn.TILE_LAYOUT, dev, DRAM) for _ in range(3)]
        for dt in (B16, B8)}
    PL = {dt: ttnn.allocate_tensor_on_device(
        ttnn.Shape([S * S, W_QKV]), dt, ttnn.TILE_LAYOUT, dev, DRAM) for dt in (B16, B8)}

    def nbytes(xdt, wdt, odt):
        return (WIDTH[xdt] * S * S * C + WIDTH[wdt] * C * W_QKV
                + WIDTH[odt] * S * S * W_QKV)

    def hm(xdt, wdt, odt):
        def f():
            G.generic_minimal_matmul(dev, X3[xdt], Wq[wdt], HM[odt], cfg, CK, DEFS,
                                     Q.KERNEL_DIR)
        return f, nbytes(xdt, wdt, odt)

    def plain(xdt, wdt, odt):
        def f():
            G.generic_minimal_matmul(dev, X2[xdt], Wq[wdt], PL[odt], cfg, CK)
        return f, nbytes(xdt, wdt, odt)

    ARMS = [
        ("hm_b16",) + hm(B16, B16, B16),
        ("plain_b16",) + plain(B16, B16, B16),
        ("hm_b8dest",) + hm(B16, B16, B8),
        ("plain_b8dest",) + plain(B16, B16, B8),
        ("hm_b8all",) + hm(B8, B8, B8),
        ("plain_b8all",) + plain(B8, B8, B8),
        ("hm_b16_aa",) + hm(B16, B16, B16),
        ("plain_b16_aa",) + plain(B16, B16, B16),
    ]
    held, sampler = force_clock()
    run_arms(ARMS)
    res["clock"] = sampler.stop()
    res["aiclk_after"] = {nd: clk.aiclk(nd) for nd in held}
    res["aa_floor_pct"] = {}
    for fam in ("hm", "plain"):
        b = res["arms"].get(fam + "_b16", {}).get("ms_med")
        a = res["arms"].get(fam + "_b16_aa", {}).get("ms_med")
        if b and a:
            res["aa_floor_pct"][fam] = abs(a - b) / b * 100
    res["ratios"] = {}
    for fam in ("hm", "plain"):
        b = res["arms"].get(fam + "_b16", {}).get("ms_med")
        for suf in ("b8dest", "b8all"):
            k = "%s_%s" % (fam, suf)
            if k in res["arms"] and b:
                r = res["arms"][k]["ms_med"] / b
                br = res["arms"][k]["bytes"] / res["arms"][fam + "_b16"]["bytes"]
                res["ratios"][k] = {"time": r, "byte_ratio": br,
                                    "realization": (1 - r) / (1 - br)}
    res["kernel_path"] = {
        "hm": ("dm_in0_sender.cpp + dm_in1_sender_out.cpp through ttnn.generic_op, "
               "HEAD_MAJOR_MT define, three [S,H,S,D] DRAM destinations"),
        "plain": ("the same generic_minimal_matmul with no defines and no kernel_dir, "
                  "one contiguous [S*S, W_QKV] DRAM destination"),
        "held_fixed": "in0, in1, block entry %s, grid %s, output elements %d, all preallocated"
                      % (list(blk or ()), list(GRID), S * S * W_QKV),
    }
    finish()

# ---------------------------------------------------------------------------- acc
else:
    # float64 reference, built from what the DEVICE holds so no arm is charged for the bf16
    # rounding of its own operands. The head-major transform moves no element inside a tile, so
    # the reference is the plain matmul reshaped to [S, H, S, D] per output.
    S, C, H, D = A.acc_m, A.cz, A.heads, A.head_dim
    W_QKV = 3 * H * D
    xt = torch.randn(S, S, C, dtype=torch.float32)
    wt = torch.randn(C, W_QKV, dtype=torch.float32) * 0.05
    res.update({"acc_n": S, "w_qkv": [C, W_QKV]})

    def score(name, qkv, ref3):
        errs = []
        for i, o in enumerate(qkv):
            d = ttnn.to_torch(o).double() - ref3[i]
            errs.append((float(d.abs().max()),
                         float(d.pow(2).mean().sqrt() / ref3[i].pow(2).mean().sqrt())))
        res["arms"][name] = {"max_abs": max(e[0] for e in errs),
                             "rel_rmse": max(e[1] for e in errs),
                             "per_output": errs, "dtype": str(qkv[0].dtype)}

    for name, xdt, wdt, odt in (("our_b16", B16, B16, B16),
                                ("our_b8dest", B16, B16, B8),
                                ("our_b8act", B8, B16, B8),
                                ("our_b8all", B8, B8, B8)):
        x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=xdt,
                            memory_config=DRAM)
        w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt,
                            memory_config=DRAM)
        xd = ttnn.to_torch(x).double().reshape(S * S, C)
        wd = ttnn.to_torch(w).double()
        full = (xd @ wd).reshape(S, S, 3, H, D)
        ref3 = [full[:, :, i].permute(0, 2, 1, 3).contiguous() for i in range(3)]
        try:
            o = Q.qkv_heads(x, w, CKC, H, D, odt, object())
            if o is None:
                res["errors"][name] = "declined: %r" % (dict(Q.REJECTS),)
            else:
                ttnn.synchronize_device(dev)
                score(name, o, ref3)
                for y in o:
                    ttnn.deallocate(y)
        except Exception as e:                                            # noqa: BLE001
            res["errors"][name] = "%s: %s" % (type(e).__name__, str(e)[:400])
        ttnn.deallocate(x)
        ttnn.deallocate(w)

    # upstream's own bfp8 pack on the same operands, as the cross-check that our packer is not
    # doing something different from ttnn's
    x = ttnn.from_torch(xt.reshape(S * S, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=B16,
                        memory_config=DRAM)
    w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=B16, memory_config=DRAM)
    ref2 = ttnn.to_torch(x).double() @ ttnn.to_torch(w).double()
    for name, odt in (("ttnn_b16", B16), ("ttnn_b8", B8)):
        y = ttnn.linear(x, w, compute_kernel_config=CKC, dtype=odt, memory_config=DRAM)
        d = ttnn.to_torch(y).double() - ref2
        res["arms"][name] = {"max_abs": float(d.abs().max()),
                             "rel_rmse": float(d.pow(2).mean().sqrt() / ref2.pow(2).mean().sqrt()),
                             "dtype": str(y.dtype)}
        ttnn.deallocate(y)

    b16 = res["arms"].get("our_b16", {}).get("rel_rmse")
    res["ratios"] = {}
    for k in ("our_b8dest", "our_b8act", "our_b8all", "ttnn_b8"):
        v = res["arms"].get(k, {}).get("rel_rmse")
        if b16 and v:
            res["ratios"][k + "_over_our_b16"] = v / b16
    t8 = res["arms"].get("ttnn_b8", {}).get("rel_rmse")
    o8 = res["arms"].get("our_b8dest", {}).get("rel_rmse")
    if t8 and o8:
        res["ratios"]["ours_over_upstream_b8"] = o8 / t8
    finish()
