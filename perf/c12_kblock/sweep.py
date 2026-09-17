#!/usr/bin/env python3
"""Does raising the inner K block buy a RATE on the census's top thin-K `linear` shapes?

The `linear` class is 42.8 % of the device term at 512 aa (108,608 calls, 4.6604 s) and sits at
21.62 TFLOP/s and 159.7 GB/s -- neither roof binding, 4,021 Mcycles above roof. That is the
signature of poor reuse, and the inner K block (`in0_block_w`) is the parameter that decides how
much arithmetic one loaded in0 block feeds. This sweep asks, per census key, whether any legal K
block beats the shipped call's rate by more than the sweep's own noise.

What the shipped call actually is, per key, is the first thing this measures. The five keys that
carry the class are plain `ttnn.linear(core_grid=CORE_GRID_MAIN)` with NO program config, so their
K block is whatever ttnn v0.68.0 derives in `create_matmul_program_config`
(`matmul_program_config.cpp:361` on the narrow 1D path, `:539` on the 2D path). `--probe` prints
that derivation next to the measured `prod` arm; the `cg_equiv` arm is the control for it.

Instrument controls, because four prior campaigns here were derailed by an uncalibrated one:
  * FLOPs and minimum bytes are counted twice, once from the logical dims and once from the tile
    grid, and the sweep refuses to report if the two disagree.
  * the compute and DRAM roofs are MEASURED in this same process on this card, and any arm whose
    rate exceeds its roof is flagged UNFIT rather than reported as a win.
  * every arm is checked against a float64 reference of its own bf16 operands, so a config that
    computes the wrong thing cannot look like a fast one.
  * arms are interleaved rep by rep (compile/warm-up bias otherwise lands entirely on arm 1), and
    the AICLK is forced and sampled at 1 kHz DURING every timed interval.

Run (card 1 on qb2):
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:c12-kblock-unlock \
  PYTHONPATH=$PWD python3 perf/c12_kblock/sweep.py --out perf/c12_kblock/sweep_512_qb2c1.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clk  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
T = 32

# (key, a_shape, w_shape, act_mc, out_mc, activation, calls@512aa, census_s, census_Mc, site)
# Shapes, call counts and seconds are the c12-linear-fusion-census table (sweep2 budget, 1350 MHz);
# the 17,920-call `1x16x512x512 K=128` key is fc1 + fc2, so it is carried here as two 8,960-call
# rows that differ only in fc1's fused silu.
SHAPES = [
    ("pair_tr_fc1", (1, 16, 512, 128), (128, 512), "L1", "L1", "silu", 8960, 0.5836, 787.8,
     "Transition.swiglu fc1 tenstorrent.py:8211"),
    ("pair_tr_fc2", (1, 16, 512, 128), (128, 512), "L1", "L1", None, 8960, 0.5836, 787.8,
     "Transition.swiglu fc2 tenstorrent.py:8222"),
    ("pair_tr_fc3", (1, 16, 512, 512), (512, 128), "L1", "DRAM", None, 8960, 0.7920, 1069.3,
     "Transition.swiglu fc3 tenstorrent.py:8233"),
    ("dit_s_768x768", (1, 512, 768), (768, 768), "DRAM", "DRAM", None, 38600, 0.5658, 763.9,
     "six DiT sites, AdaLN :9395/:9403 + AttentionPairBias :8159/:8169 + DiT :9610/:9519"),
    ("ctb_768x1536", (1, 512, 768), (768, 1536), "DRAM", "DRAM", None, 15200, 0.3418, 461.5,
     "ConditionedTransitionBlock :9492/:9498/:9506"),
    ("trimul_p_out", (1, 512, 512, 128), (128, 128), "DRAM", "DRAM", None, 560, 0.5276, 712.3,
     "trimul p_out :6789 via _pair_proj_linear -- already carries our tuned config; the L1 "
     "destination production prefers needs its own rung ladder, so this row is the DRAM fallback"),
]
MC = {"L1": L1, "DRAM": DRAM}


def med(xs):
    return sorted(xs)[len(xs) // 2]


def divisors(n, cap=None):
    return [d for d in range(1, n + 1) if n % d == 0 and (cap is None or d <= cap)]


def counts(a_shape, w_shape):
    """FLOPs and minimum bytes, counted from the logical dims AND from the tile grid.

    A dense matmul has an exact FLOP count and an exact minimum byte count. Two independent counts
    that agree is the cheapest calibration this sweep can carry; if they ever disagree the shape is
    not tile-aligned and every rate below it would be wrong.
    """
    b = math.prod(a_shape[:-2])
    m, k = a_shape[-2], a_shape[-1]
    n = w_shape[-1]
    flops = 2 * b * m * n * k
    tile_flops = 2 * (b * (m // T) * (k // T) * (n // T)) * T ** 3
    rd = 2 * (b * m * k + k * n)
    wr = 2 * b * m * n
    tile_bytes = 2 * T * T * (b * (m // T) * (k // T) + (k // T) * (n // T) + b * (m // T) * (n // T))
    fit = (m % T == 0 and k % T == 0 and n % T == 0
           and flops == tile_flops and rd + wr == tile_bytes)
    return {"flops": flops, "read_B": rd, "write_B": wr, "bytes": rd + wr, "fit": fit,
            "b": b, "m": m, "k": k, "n": n, "mt": m // T, "kt": k // T, "nt": n // T,
            "mt_total": b * m // T}


def roofs(dev, ckc, node):
    """Compute and DRAM roofs, measured here. A roof that is asserted rather than measured is how
    a campaign talks itself into a win that is not there."""
    out = {}
    n = 4096
    a = ttnn.from_torch(torch.randn(1, 1, n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    b = ttnn.from_torch(torch.randn(1, 1, n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    s = clk.Sampler(node)
    s.start()
    ms = timed_one(dev, lambda: ttnn.deallocate(ttnn.matmul(
        a, b, memory_config=DRAM, compute_kernel_config=ckc)), pipe=4, reps=5)
    out["clk_compute"] = s.stop()
    out["compute_TFLOPs"] = round(2 * n ** 3 / 1e9 / ms, 2)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    nb = 32 * 1024 * 1024
    rows = nb // (2 * 1024)
    t = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    s = clk.Sampler(node)
    s.start()
    ms = timed_one(dev, lambda: ttnn.deallocate(ttnn.clone(t, memory_config=DRAM)), pipe=4, reps=5)
    out["clk_dram"] = s.stop()
    out["dram_rw_GBs"] = round(2 * nb / 1e9 / (ms / 1e3), 1)
    ttnn.deallocate(t)
    return out


def timed_one(dev, fn, pipe=4, reps=5):
    for _ in range(2):
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


def timed_interleaved(dev, arms, pipe=4, reps=5, warm=2):
    """Median ms/call per arm, arms interleaved rep by rep.

    Measuring arm A to completion and then arm B puts all of the compile, cache-warm and
    governor-ramp bias on A. The fleet has made that mistake before, so the rep loop is outside the
    arm loop here and the warm-up is done for every arm before any arm is timed.
    """
    for _, fn in arms:
        for _ in range(warm):
            fn()
    ttnn.synchronize_device(dev)
    acc = {name: [] for name, _ in arms}
    for _ in range(reps):
        for name, fn in arms:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(pipe):
                fn()
            ttnn.synchronize_device(dev)
            acc[name].append((time.perf_counter() - t0) * 1e3 / pipe)
    return {k: {"ms": med(v), "ms_all": [round(x, 4) for x in v]} for k, v in acc.items()}


def derived_incumbent(c, nc, gx, gy, all_dram):
    """What ttnn v0.68.0 derives for `ttnn.linear(core_grid=...)` at this shape.

    `create_matmul_program_config`, matmul_program_config.cpp: narrow (height/width ratio > 8, or
    all-DRAM with a dim inside one tile) takes the 1D systolic factory, whose K block is
    `K / 32 % 2 == 0 ? 2 : 1` (:361); everything else takes the 2D factory, whose K block starts at
    4 and walks down to the first divisor of Kt (:539-541). Neither is a function of the shape's
    arithmetic intensity, and neither is ever the whole contraction above Kt = 4.
    """
    height, width = c["mt_total"] * T, c["nt"] * T
    ratio = max(height, width) / min(height, width)
    narrow = ratio > 8 or (all_dram and min(height, width) <= T)
    if narrow:
        bw = 2 if c["kt"] % 2 == 0 else 1
        return {"family": "1d", "in0_block_w": bw, "per_core_M": -(-c["mt_total"] // nc),
                "per_core_N": c["nt"], "narrow_ratio": round(ratio, 1)}
    bw = 4
    while c["kt"] % bw:
        bw -= 1
    return {"family": "2d", "in0_block_w": bw, "per_core_M": -(-c["mt_total"] // gy),
            "per_core_N": -(-c["nt"] // gx), "narrow_ratio": round(ratio, 1)}


def subblocks(obh, obw):
    sh = max((h for h in range(min(4, obh), 0, -1) if obh % h == 0), default=1)
    sw = max((w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0), default=1)
    return sh, sw


def make_cfg(family, gx, gy, bw, pcm, pcn, obh, obw, act):
    sh, sw = subblocks(obh, obw)
    if family == "1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sh,
            out_subblock_w=sw, out_block_h=obh, out_block_w=obw, per_core_M=pcm, per_core_N=pcn,
            fuse_batch=True, fused_activation=act, mcast_in0=False)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=obh, out_block_w=obw, per_core_M=pcm, per_core_N=pcn,
        transpose_mcast=False, fused_activation=act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pipe", type=int, default=4)
    ap.add_argument("--probe", action="store_true", help="print the derivation and stop")
    a = ap.parse_args()

    import tt_bio.tenstorrent as TB
    print(f"tt_bio from {TB.__file__}", flush=True)
    dev = TB.get_device()
    nodes = clk.nodes_open_by_this_process()
    held = clk.force(a.mhz, nodes) if a.mhz else []
    node = nodes[0]
    ag = dev.compute_with_storage_grid_size()
    gx, gy = int(ag.x), int(ag.y)
    nc = gx * gy
    grid = ttnn.CoreGrid(y=gy, x=gx)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    print(f"grid {gx}x{gy} = {nc} cores | nodes {nodes} | clock forced {a.mhz} on {held} "
          f"| aiclk now {clk.aiclk(node)} MHz", flush=True)

    R = roofs(dev, ckc, node) if not a.probe else {}
    if R:
        print("ROOFS " + json.dumps(R), flush=True)

    torch.manual_seed(0)
    recs = []
    for (key, a_shape, w_shape, amc, omc, act, calls, cs, cmc, site) in SHAPES:
        if a.only and a.only not in key:
            continue
        c = counts(a_shape, w_shape)
        all_dram = amc == "DRAM" and omc == "DRAM"
        inc = derived_incumbent(c, nc, gx, gy, all_dram)
        print(f"\n=== {key}  a{list(a_shape)} @ w{list(w_shape)}  "
              f"mt_total={c['mt_total']} kt={c['kt']} nt={c['nt']}  act={act}  {amc}->{omc}\n"
              f"    census {calls} calls, {cs:.4f} s, {cmc:.1f} Mc | {site}\n"
              f"    ttnn v0.68.0 would derive: {inc}", flush=True)
        if not c["fit"]:
            print("    UNFIT: the two FLOP/byte counts disagree, skipping", flush=True)
            continue
        if a.probe:
            recs.append({"key": key, "counts": c, "derived": inc})
            continue

        at = torch.randn(*a_shape) * 0.1
        wt = torch.randn(*w_shape) * 0.1
        ta = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                             memory_config=MC[amc])
        tw = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                             memory_config=DRAM)
        # The reference is float64 over the operands the device actually holds, so an arm is scored
        # against arithmetic and not against another approximation.
        ab = ttnn.to_torch(ta).double()
        wb = ttnn.to_torch(tw).double()
        ref = ab @ wb
        if act == "silu":
            ref = ref * torch.sigmoid(ref)
        gflop = c["flops"] / 1e9
        gb = c["bytes"] / 1e9

        arms, cfgs = [], {}
        arms.append(("prod", lambda: ttnn.deallocate(ttnn.linear(
            ta, tw, compute_kernel_config=ckc, core_grid=grid, memory_config=MC[omc],
            activation=act))))
        pcm_1d = -(-c["mt_total"] // nc)
        pcm_2d = -(-c["mt_total"] // gy)
        pcn_2d = -(-c["nt"] // gx)
        fam = inc["family"]
        pcm, pcn = (pcm_1d, c["nt"]) if fam == "1d" else (pcm_2d, pcn_2d)
        fused_act = None
        if act == "silu":
            try:
                fused_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
            except Exception as e:
                print(f"    fused silu not constructible ({type(e).__name__}); this key's "
                      f"explicit arms would not be the same op, so they are skipped. fc2 carries "
                      f"the identical shape without the silu.", flush=True)
                fused_act = "UNAVAILABLE"
        # Bounded arm set: 1, 2, whatever ttnn derives, 4, 8, half of K and the whole of K --
        # every rung that could change the reuse story, without paying compile for all of them.
        bw_set = sorted({1, 2, 4, 8, inc["in0_block_w"], c["kt"] // 2, c["kt"]}
                        & set(divisors(c["kt"])))
        if fused_act == "UNAVAILABLE":
            bw_set = []
        for bw in bw_set:
            for obh in sorted(set(divisors(pcm)), reverse=True)[:2]:
                for obw in sorted({pcn, min(pcn, 16), min(pcn, 8)} & set(divisors(pcn)),
                                  reverse=True):
                    name = f"{fam}_bw{bw}_obh{obh}_obw{obw}"
                    if name in cfgs:
                        continue
                    try:
                        cfg = make_cfg(fam, gx, gy, bw, pcm, pcn, obh, obw, fused_act)
                    except Exception as e:
                        print(f"    {name:26s} CFG-REFUSED {type(e).__name__}", flush=True)
                        continue
                    cfgs[name] = cfg
                    arms.append((name, (lambda cfg=cfg: ttnn.deallocate(ttnn.linear(
                        ta, tw, compute_kernel_config=ckc, memory_config=MC[omc],
                        program_config=cfg)))))

        # Drop any arm the device refuses outright, before timing, so a refusal is not a fast arm.
        live, out = [], {}
        for name, fn in arms:
            try:
                fn()
                ttnn.synchronize_device(dev)
            except Exception as e:
                print(f"    {name:26s} SKIP {type(e).__name__}: {str(e)[:70]}", flush=True)
                continue
            live.append((name, fn))
        s = clk.Sampler(node)
        s.start()
        res = timed_interleaved(dev, live, pipe=a.pipe, reps=a.reps)
        clock = s.stop()
        print(f"    clock during this key: {clock}", flush=True)

        prod_t = None
        for name, _ in live:
            ms = res[name]["ms"]
            if name == "prod":
                y = ttnn.linear(ta, tw, compute_kernel_config=ckc, core_grid=grid,
                                memory_config=MC[omc], activation=act)
            else:
                y = ttnn.linear(ta, tw, compute_kernel_config=ckc, memory_config=MC[omc],
                                program_config=cfgs[name])
            yt = ttnn.to_torch(y).double()
            ttnn.deallocate(y)
            err = (yt - ref).abs().max().item()
            rms = (yt - ref).pow(2).mean().sqrt().item()
            tf = gflop / ms
            gbs = gb / (ms / 1e3)
            unfit = bool(R) and (tf > 1.05 * R["compute_TFLOPs"] or gbs > 1.05 * R["dram_rw_GBs"])
            rec = {"ms": round(ms, 5), "TFLOPs": round(tf, 2), "GBs": round(gbs, 1),
                   "max_abs_vs_f64": err, "rmsd_vs_f64": rms, "unfit": unfit,
                   "ms_all": res[name]["ms_all"]}
            if name == "prod":
                prod_t = ms
                rec["derived"] = inc
            elif prod_t is None:
                rec["x_vs_prod"] = None      # the shipped call itself was refused at this shape
            else:
                rec["x_vs_prod"] = round(prod_t / ms, 4)
                rec["fold_s_saved"] = round(cs * (1 - ms / prod_t), 4)
                rec["fold_Mc_saved"] = round(cmc * (1 - ms / prod_t), 1)
            out[name] = rec
            flag = " UNFIT" if unfit else ""
            extra = ("" if name == "prod" or rec.get("x_vs_prod") is None
                     else f"  {rec['x_vs_prod']:.4f}x  {rec['fold_s_saved']:+.4f} s fold")
            print(f"    {name:26s} {ms:8.4f} ms  {tf:7.2f} TF/s  {gbs:7.1f} GB/s  "
                  f"maxabs {err:.4f}{extra}{flag}", flush=True)

        cand = {k: v for k, v in out.items()
                if k != "prod" and not v["unfit"] and v.get("x_vs_prod") is not None}
        best = min(cand, key=lambda k: cand[k]["ms"]) if cand else None
        rec = {"key": key, "a_shape": list(a_shape), "w_shape": list(w_shape), "act": act,
               "amc": amc, "omc": omc, "site": site, "calls": calls, "census_s": cs,
               "census_Mc": cmc, "counts": c, "derived": inc, "clock": clock, "arms": out}
        if best:
            rec["best"] = {"arm": best, **cand[best]}
            print(f"    BEST {best}  {cand[best]['x_vs_prod']:.4f}x vs prod  "
                  f"{cand[best]['fold_s_saved']:+.4f} s / {cand[best]['fold_Mc_saved']:+.1f} Mc "
                  f"of fold", flush=True)
        recs.append(rec)
        ttnn.deallocate(ta)
        ttnn.deallocate(tw)

    payload = {"clock_target_MHz": a.mhz, "grid": [gx, gy], "cores": nc, "roofs": R,
               "nodes": nodes, "pipe": a.pipe, "reps": a.reps, "shapes": recs,
               "ttnn": getattr(ttnn, "__version__", "0.68.0")}
    if a.out:
        json.dump(payload, open(a.out, "w"), indent=1)
        print("\nwrote", a.out, flush=True)
    tot = sum(r["best"]["fold_s_saved"] for r in recs if r.get("best")
              and r["best"]["fold_s_saved"] > 0)
    print(f"TOTAL pre-registered fold seconds from the K block alone: {tot:+.4f} s", flush=True)
    clk.release()


if __name__ == "__main__":
    main()
