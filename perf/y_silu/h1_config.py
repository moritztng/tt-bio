#!/usr/bin/env python3
"""H1 -- is the fused-activation penalty ARITHMETIC or program-config SELECTION?

Same explicit program config run twice, `fused_activation` set and unset, every other field pinned
identical, on the fold's own tensor `[1, 30, 298, 256] x [256, 1024]` (censused from a live fold).
Plus two discriminators the brief did not ask for and that separate the two mechanisms directly:

  * fused RELU / GELU / SQRT against fused SILU. RELU is one SFPU instruction, SILU is a
    transcendental. If every fused activation costs the same, the penalty is structural (the
    activation changes the config or the kernel phase) and not silu's own arithmetic.
  * the standalone SFPU roof for each of the same activations, so each fused penalty can be scored
    against its own standalone cost rather than against silu's.

    TT_VISIBLE_DEVICES=0 python3 perf/y_silu/h1_config.py --out perf/y_silu/h1.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import numpy as np, torch, ttnn


def load():
    return [round(v, 2) for v in os.getloadavg()]


def med(v):
    return sorted(v)[len(v) // 2]


def timed(dev, fn, k, reps=7, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(k):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t) / k * 1e6)
    return round(med(out), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "h1.json"))
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    dev = T.get_device()
    L1 = ttnn.L1_MEMORY_CONFIG
    CG = T.CORE_GRID_MAIN
    gx, gy = T.COMPUTE_GRID_MAIN
    res = dict(load_start=load(), grid=[gx, gy])

    # production, protenix.py:1609 -- HiFi4, fp32 dest accumulate, packer L1 accumulate
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    res["ckc"] = dict(math_fidelity="HiFi4", fp32_dest_acc_en=True, packer_l1_acc=True,
                      source="tt_bio/protenix.py:1609")

    torch.manual_seed(0)
    xt = torch.randn(1, 30, 298, 256, dtype=torch.bfloat16)
    wt = torch.randn(256, 1024, dtype=torch.bfloat16) * 0.05
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)

    # ---- what does the auto path pick, and how does it respond to the activation? --------------
    auto = {}
    for act in (None, "relu", "gelu", "silu"):
        def fn(act=act):
            y = ttnn.linear(x, w, activation=act, compute_kernel_config=ckc, memory_config=L1,
                            dtype=ttnn.bfloat16, core_grid=CG)
            ttnn.deallocate(y)
        auto[str(act)] = timed(dev, fn, a.k)
        print("auto", act, auto[str(act)], flush=True)
    res["auto_by_activation"] = auto

    # ---- standalone SFPU cost of each activation, same output shape ------------------------------
    y0 = ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=L1, dtype=ttnn.bfloat16,
                     core_grid=CG)
    ttnn.synchronize_device(dev)
    sfpu = {}
    for name, op in (("relu", ttnn.relu), ("gelu", ttnn.gelu), ("silu", ttnn.silu),
                     ("clone", ttnn.clone)):
        def fn(op=op):
            z = op(y0, memory_config=L1); ttnn.deallocate(z)
        sfpu[name] = timed(dev, fn, max(4, a.k // 2))
        print("standalone", name, sfpu[name], flush=True)
    res["standalone_sfpu"] = sfpu

    # ---- the same EXPLICIT config, activation set and unset ---------------------------------------
    UWP = ttnn.UnaryWithParam
    SILU = UWP(ttnn.UnaryOpType.SILU)
    Mt_total, Kt, Nt = 30 * 10, 8, 32     # 298 -> 320 rows = 10 tiles per batch element
    res["tiles"] = dict(Mt_total=Mt_total, Kt=Kt, Nt=Nt)
    cands = []
    for pcM in (30, 15):
        for pcN in (3, 4):
            for bw in (8, 4):
                for sh, sw in ((1, 3), (3, 1), (2, 1), (1, 1), (4, 1), (2, 2), (1, 4), (1, 2)):
                    if pcM % sh or pcN % sw or sh * sw > 4:
                        continue
                    cands.append((pcM, pcN, bw, sh, sw))
    seen, rows = set(), []
    for pcM, pcN, bw, sh, sw in cands:
        key = (pcM, pcN, bw, sh, sw)
        if key in seen:
            continue
        seen.add(key)
        row = dict(per_core_M=pcM, per_core_N=pcN, in0_block_w=bw, out_subblock_h=sh,
                   out_subblock_w=sw)
        try:
            def mk(fa):
                return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
                    out_subblock_h=sh, out_subblock_w=sw, per_core_M=pcM, per_core_N=pcN,
                    transpose_mcast=False, fused_activation=fa)
            for label, fa in (("bare", None), ("silu", SILU)):
                cfg = mk(fa)
                def fn(cfg=cfg):
                    y = ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=L1,
                                    dtype=ttnn.bfloat16, program_config=cfg)
                    ttnn.deallocate(y)
                row[label] = timed(dev, fn, max(4, a.k // 2), reps=5)
            row["delta"] = round(row["silu"] - row["bare"], 3)
            print("explicit", row, flush=True)
        except Exception as e:
            row["error"] = repr(e)[:180]
        rows.append(row)
        with open(a.out, "w") as f:
            json.dump(dict(res, explicit=rows), f, indent=1)
    res["explicit"] = rows
    res["load_end"] = load()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
