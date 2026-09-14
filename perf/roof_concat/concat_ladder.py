#!/usr/bin/env python3
"""The one-op head re-assembly, priced on the shipped DiffusionTransformerLayer, per rung.

Two layers built from the SAME torch weights in one process -- one with
TT_BIO_APB_CONCAT_HEADS off, one with it on -- then run against the same activations at
298 / 512 / 768 / 1024 aa. The fused arm keeps the pad lanes, so its gate and output
projection are n_heads*padded_head_dim wide; that cost grows with the token axis, which is
why every rung is measured and none is extrapolated.

Method, unchanged from perf/roof_difftx/dit_layer.py: one process, one device, one session,
`reps` enqueued per synchronize, the whole arm set run once per block in a FIXED order so a
JIT warm-up or a clock ramp cannot bias one arm against another, minimum over blocks. The
shipped layer is entered twice under two names to carry an own-session A/A floor, and the
dense cube runs in the same session so the roof is measured here and not asserted.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import ttnn

import tt_bio.tenstorrent as T

HERE = Path(__file__).resolve().parent


def weights(dim, inner, scale=0.02):
    def w(*shape):
        return torch.randn(*shape, dtype=torch.float32) * scale
    d = {}
    for pfx in ("adaln", "transition.adaln"):
        d[f"{pfx}.s_norm.weight"] = torch.ones(dim)
        d[f"{pfx}.s_scale.weight"] = w(dim, dim)
        d[f"{pfx}.s_scale.bias"] = w(dim)
        d[f"{pfx}.s_bias.weight"] = w(dim, dim)
    for k in ("proj_q", "proj_k", "proj_v", "proj_g", "proj_o"):
        d[f"pair_bias_attn.{k}.weight"] = w(dim, dim)
    d["pair_bias_attn.proj_q.bias"] = w(dim)
    d["output_projection_linear.weight"] = w(dim, dim)
    d["output_projection_linear.bias"] = w(dim)
    d["transition.swish_gate.0.weight"] = w(2 * inner, dim)
    d["transition.a_to_b.weight"] = w(inner, dim)
    d["transition.b_to_a.weight"] = w(dim, inner)
    d["transition.output_projection.0.weight"] = w(dim, dim)
    d["transition.output_projection.0.bias"] = w(dim)
    return d


def mm(m, k, n):
    return 2 * m * k * n


def bench(arms, order, blocks, dev, warm=2):
    err, live = {}, []
    for n in order:
        try:
            for _ in range(warm):
                ttnn.deallocate(arms[n][0]())
            live.append(n)
        except Exception as e:                                                # noqa: BLE001
            err[n] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:240])
            print("SKIP %-18s %s" % (n, err[n]), flush=True)
    ttnn.synchronize_device(dev)
    best = {n: None for n in live}
    for _ in range(blocks):
        for n in list(best):
            fn, _f, reps = arms[n]
            outs = []
            try:
                t0 = time.perf_counter()
                for _ in range(reps):
                    outs.append(fn())
                    if len(outs) > 4:
                        ttnn.deallocate(outs.pop(0))
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / reps
            except Exception as e:                                            # noqa: BLE001
                err[n] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:240])
                print("DROP %-18s %s" % (n, err[n]), flush=True)
                best.pop(n, None)
                for o in outs:
                    ttnn.deallocate(o)
                ttnn.synchronize_device(dev)
                continue
            for o in outs:
                ttnn.deallocate(o)
            best[n] = dt if best[n] is None else min(best[n], dt)
    return {n: v for n, v in best.items() if v is not None}, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "ladder.json")
    ap.add_argument("--seq", type=int, nargs="+", default=[320, 512, 768, 1024])
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    dev = T.get_device()
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    dim, H = a.dim, a.heads
    hd, inner = dim // H, 2 * dim
    torch.manual_seed(0)
    wd = weights(dim, inner)

    mods = {}
    for name, on in (("ship", False), ("concat", True)):
        T._APB_CONCAT_HEADS = on
        mods[name] = T.DiffusionTransformerLayer(dim, H, False, dict(wd), kc)
    assert not mods["ship"].attn_pair_bias._concat_heads
    assert mods["concat"].attn_pair_bias._concat_heads
    phd = mods["concat"].attn_pair_bias.padded_head_dim

    cc = dev.compute_with_storage_grid_size()
    out = {"host": platform.node(), "arch": str(dev.arch()), "grid": [cc.x, cc.y],
           "core_grid_main": [T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y],
           "dim": dim, "heads": H, "head_dim": hd, "padded_head_dim": phd,
           "blocks": a.blocks, "reps": a.reps,
           "loadavg_start": open("/proc/loadavg").read().split()[:3], "rungs": []}

    def t(shape, sc=1.0):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16) * sc,
                               layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # the roof, measured in this session
    c4, c4b = t((4096, 4096)), t((4096, 4096))
    cube = {"cube4096": (lambda: ttnn.matmul(c4, c4b, compute_kernel_config=kc,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG),
                         mm(4096, 4096, 4096), 3)}
    cube["cube4096_AA"] = cube["cube4096"]
    cb, cerr = bench(cube, ["cube4096", "cube4096_AA"], a.blocks, dev)
    out["cube4096_TFLOPs"] = mm(4096, 4096, 4096) / cb["cube4096"] / 1e12
    out["cube_AA_pct"] = 100 * abs(cb["cube4096_AA"] - cb["cube4096"]) / cb["cube4096"]
    print("cube %.2f TFLOP/s  A/A %.3f %%" % (out["cube4096_TFLOPs"], out["cube_AA_pct"]),
          flush=True)
    ttnn.deallocate(c4); ttnn.deallocate(c4b)

    F_ADALN = 2 * mm(1, dim, dim)
    for S in a.seq:
        aa, ss = t((1, S, dim)), t((1, S, dim))
        zz = t((1, H, S, S), 0.1)
        oh = t((1, H, S, phd))
        o_nar, o_pad = t((1, S, H * hd)), t((1, S, H * phd))
        keep = [aa, ss, zz, oh, o_nar, o_pad]
        f_adaln = 2 * mm(S, dim, dim)
        f_attn = (mm(S, dim, 3 * dim) + 2 * mm(S, dim, dim) + 2 * H * mm(S, S, hd))
        f_layer = (f_adaln + f_attn + mm(S, dim, dim)
                   + f_adaln + 3 * mm(S, dim, inner) + mm(S, inner, dim) + mm(S, dim, dim))
        reps = max(2, a.reps if S <= 512 else a.reps // 2)

        def epi_ship():
            o = oh[:, :, :, :hd]
            o = ttnn.permute(o, (0, 1, 3, 2))
            o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
            return ttnn.permute(o, (0, 2, 1))

        def epi_cat():
            o = ttnn.experimental.nlp_concat_heads(oh)
            return ttnn.reshape(o, (1, S, H * phd))

        arms = {
            "layer_ship": (lambda: mods["ship"](aa, ss, zz), f_layer, reps),
            "layer_concat": (lambda: mods["concat"](aa, ss, zz), f_layer, reps),
            "epi_ship": (epi_ship, 0, reps * 2),
            "epi_cat": (epi_cat, 0, reps * 2),
            "gate_narrow": (lambda: ttnn.linear(aa, mods["ship"].attn_pair_bias.g_weight,
                                                compute_kernel_config=kc,
                                                core_grid=T.CORE_GRID_MAIN),
                            mm(S, dim, H * hd), reps * 2),
            "gate_wide": (lambda: ttnn.linear(aa, mods["concat"].attn_pair_bias.g_weight,
                                              compute_kernel_config=kc,
                                              core_grid=T.CORE_GRID_MAIN),
                          mm(S, dim, H * phd), reps * 2),
            "out_narrow": (lambda: ttnn.linear(o_nar, mods["ship"].attn_pair_bias.o_weight,
                                               compute_kernel_config=kc,
                                               core_grid=T.CORE_GRID_MAIN),
                           mm(S, H * hd, dim), reps * 2),
            "out_wide": (lambda: ttnn.linear(o_pad, mods["concat"].attn_pair_bias.o_weight,
                                             compute_kernel_config=kc,
                                             core_grid=T.CORE_GRID_MAIN),
                         mm(S, H * phd, dim), reps * 2),
        }
        arms["layer_ship_AA"] = arms["layer_ship"]
        order = ["layer_ship", "layer_concat", "epi_ship", "epi_cat", "gate_narrow",
                 "gate_wide", "out_narrow", "out_wide", "layer_ship_AA"]
        best, err = bench(arms, order, a.blocks, dev)
        rec = {"S": S, "aa_equiv": S, "reps": reps, "refused": err,
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "ms": {n: best[n] * 1e3 for n in best},
               "F_LAYER_GFLOP": f_layer / 1e9}
        if "layer_ship" in best and "layer_ship_AA" in best:
            rec["AA_pct"] = 100 * abs(best["layer_ship_AA"] - best["layer_ship"]) / best["layer_ship"]
        if "layer_ship" in best and "layer_concat" in best:
            rec["layer_ratio"] = best["layer_ship"] / best["layer_concat"]
            rec["layer_TFLOPs_ship"] = f_layer / best["layer_ship"] / 1e12
            rec["pct_of_cube_ship"] = (100 * rec["layer_TFLOPs_ship"] / out["cube4096_TFLOPs"])
        if "epi_ship" in best and "epi_cat" in best:
            rec["epi_ratio"] = best["epi_ship"] / best["epi_cat"]
            rec["epi_saved_us"] = (best["epi_ship"] - best["epi_cat"]) * 1e6
        for pair in ("gate", "out"):
            if f"{pair}_narrow" in best and f"{pair}_wide" in best:
                rec[f"{pair}_pad_us"] = (best[f"{pair}_wide"] - best[f"{pair}_narrow"]) * 1e6
        out["rungs"].append(rec)
        print("S=%-5d ratio %s  A/A %s %%  epi %s  gate_pad %s us  out_pad %s us"
              % (S, ("%.4f" % rec["layer_ratio"]) if "layer_ratio" in rec else "-",
                 ("%.3f" % rec["AA_pct"]) if "AA_pct" in rec else "-",
                 ("%.2fx" % rec["epi_ratio"]) if "epi_ratio" in rec else "-",
                 ("%.2f" % rec.get("gate_pad_us", float("nan"))),
                 ("%.2f" % rec.get("out_pad_us", float("nan")))), flush=True)
        for x in keep:
            ttnn.deallocate(x)
        a.out.write_text(json.dumps(out, indent=1))
    out["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
