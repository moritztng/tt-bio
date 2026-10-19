"""SaProt's four projections at the page shape (batch of 7 x 544 rows): the model's own call against 2D/1D multicast
programs and minimal_matmul, each timed on device and scored against float64 of the same bf16 operands.

The 650M/1.3B census (spd-gap-whlm, WH 1000 MHz) runs them on ttnn.linear's auto pick at 26-41 TF/s against a ~56
TF/s HiFi4 roof, and fidelity moved nothing, so the program, not the math, binds. This finds the best program per
shape and arch; tt_bio.saprot picks it if it beats the model's call at equal error.

    python3 perf/spd_m_saprot/lin_sweep.py --model saprot-650m [--fast] --out lin.jsonl

--pick-only times just the model's call and the program tt_bio.saprot._saprot_prog ships for each shape (arm "pick").
"""
import argparse
import itertools
import json
import math
import time

import torch
import ttnn

MODELS = {"saprot-35m": (480, 1920), "saprot-650m": (1280, 5120), "saprot-1.3b": (1280, 5120)}


def _subblocks(pm, pn, cap):
    return [(h, w) for h in range(1, pm + 1) for w in range(1, pn + 1)
            if pm % h == 0 and pn % w == 0 and h * w <= cap]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m", choices=sorted(MODELS))
    ap.add_argument("--fast", action="store_true", help="bfloat8_b weights, as --fast loads them")
    ap.add_argument("--rows", type=int, default=7 * 544)
    ap.add_argument("--shapes", default="qkv,out,fc1,fc2")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--max-arms", type=int, default=60, help="per shape and family, best-first by a crude L1 rank")
    ap.add_argument("--bf16-acc", action="store_true", help="bf16 dest accumulation (a --fast option): 2 x 4 subblocks fit")
    ap.add_argument("--fidelity", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"], help="override the model's math fidelity")
    ap.add_argument("--pick-only", action="store_true", help="only the model call and the shipped program")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T, ops
    from tt_bio import saprot as S
    T.set_fast_mode(a.fast)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    cfg, _ = S.CONFIGS[a.model]
    ck = S.Saprot(**cfg).compute_kernel_config
    if a.bf16_acc or a.fidelity:
        ck = type(ck)(math_fidelity=getattr(ttnn.MathFidelity, a.fidelity) if a.fidelity else ck.math_fidelity,
                      math_approx_mode=ck.math_approx_mode, fp32_dest_acc_en=ck.fp32_dest_acc_en and not a.bf16_acc,
                      packer_l1_acc=ck.packer_l1_acc)
    d, ff = MODELS[a.model]
    dh = d if (d // cfg["n_heads"]) % 64 == 0 else cfg["n_heads"] * 32  # 35M pads its heads 24 -> 32
    shapes = {"qkv": (d, 3 * dh, None), "out": (dh, d, None), "fc1": (d, ff, "gelu"), "fc2": (ff, d, None),
              # the FFN split in two along its intermediate dim: a half-width fc1 fits a 2D program's fp32 output in L1
              "fc1h": (d, ff // 2, "gelu"), "fc2h": (ff // 2, d, None)}
    wdt = ttnn.bfloat8_b if a.fast else ttnn.bfloat16
    out = open(a.out, "a")
    gen = torch.Generator().manual_seed(0)
    B, L = 7, a.rows // 7
    M = a.rows
    for name in a.shapes.split(","):
        K, N, act = shapes[name]
        x = torch.randn(B, L, K, generator=gen).to(torch.bfloat16)
        w = (torch.randn(K, N, generator=gen) / math.sqrt(K)).to(torch.bfloat16)
        b = (torch.randn(N, generator=gen) * 0.1).to(torch.bfloat16)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
        wq = ttnn.to_torch(wd).to(torch.bfloat16) if a.fast else w
        ref = x.double() @ wq.double() + b.double()
        if act == "gelu":
            ref = torch.nn.functional.gelu(ref)
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        bd = ttnn.from_torch(b.reshape(1, N), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        flops = 2.0 * M * K * N

        def run(tag, fn, meta):
            try:
                o = fn(); ttnn.synchronize_device(dev)
                got = ttnn.to_torch(o).double().reshape(ref.shape); ttnn.deallocate(o)
                ts = []
                for _ in range(a.repeat):
                    t = time.perf_counter()
                    for _ in range(a.inner):
                        ttnn.deallocate(fn())
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t) / a.inner * 1e3)
                ms = sorted(ts)[len(ts) // 2]
                rel = float((got - ref).norm() / ref.norm())
                rec = dict(model=a.model, fast=a.fast, shape=name, M=M, K=K, N=N, arm=tag, ms=round(ms, 4),
                           tflops=round(flops / ms / 1e9, 1), rel_l2=rel, **meta)
            except Exception as e:  # noqa: BLE001 -- a refused program is a data point
                rec = dict(model=a.model, fast=a.fast, shape=name, arm=tag, error=str(e).splitlines()[0][:200], **meta)
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n"); out.flush()
            return rec

        kw = {"activation": act} if act else {}
        lin = S._SaprotLin.__new__(S._SaprotLin)
        lin.compute_kernel_config, lin.norm_weight = ck, torch.empty(d)
        mck = lin._ck(wd)   # what the model ships for this weight (HiFi2 at --fast, _SAPROT_HIFI2); every arm runs it
        run("model", lambda: ops.linear(xd, wd, bias=bd, compute_kernel_config=mck, dtype=ttnn.bfloat16,
                                        core_grid=T.CORE_GRID_MAIN, **kw), {})
        hifi2 = mck.math_fidelity == ttnn.MathFidelity.HiFi2
        if S._saprot_prog(xd, wd, act, hifi2) is not None:
            before = list(S.PROG_STATS)
            rec = run("pick", lambda: lin._lin(xd, wd, bias=bd, **kw), {"kind": S._saprot_prog(xd, wd, act, hifi2)[0]})
            assert S.PROG_STATS != before or "error" in rec, "the shipped program fell back to the default"
        if a.pick_only:
            ttnn.deallocate(xd); ttnn.deallocate(wd); ttnn.deallocate(bd)
            continue
        fa = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0) if act else None   # exact GELU, as the model ships it
        mt, kt, nt = -(-M // 32), K // 32, N // 32
        cap = 4 if mck.fp32_dest_acc_en else 8
        arms = []
        for gx, gy in {(g.x, g.y)}:
            for tm in (False, True):  # 2D: rows over y (or x when transposed)
                rows_ax, cols_ax = (gx, gy) if tm else (gy, gx)
                pm, pn = -(-mt // rows_ax), -(-nt // cols_ax)
                for kb in [k for k in (1, 2, 4, 5, 8, 10) if kt % k == 0]:
                    for sh, sw in _subblocks(pm, pn, cap):
                        l1 = (pm * kb + kb * pn) * 2 * 2048 + pm * pn * (4096 if mck.fp32_dest_acc_en else 2048) * 2
                        arms.append((l1, "2d", dict(tm=tm, kb=kb, sh=sh, sw=sw, pm=pm, pn=pn)))
            pn1 = -(-nt // (gx * gy))
            for kb in [k for k in (1, 2, 4, 5, 8) if kt % k == 0]:
                for sh, sw in _subblocks(mt, pn1, cap):
                    if sh * sw < 2:
                        continue
                    arms.append((mt * pn1 * 4096, "1d", dict(kb=kb, sh=sh, sw=sw, pm=mt, pn=pn1)))
        # Best-first: biggest subblock, then mid kb; keep the sweep bounded.
        arms.sort(key=lambda r: (r[1], -(r[2]["sh"] * r[2]["sw"]), abs(r[2]["kb"] - 4)))
        seen = {}
        for _, fam, p in arms:
            if seen.get(fam, 0) >= a.max_arms:
                continue
            seen[fam] = seen.get(fam, 0) + 1
            if fam == "2d":
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y), in0_block_w=p["kb"], out_subblock_h=p["sh"],
                    out_subblock_w=p["sw"], per_core_M=p["pm"], per_core_N=p["pn"], transpose_mcast=p["tm"],
                    fused_activation=fa, fuse_batch=True)
            else:
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y), in0_block_w=p["kb"], out_subblock_h=p["sh"],
                    out_subblock_w=p["sw"], per_core_M=p["pm"], per_core_N=p["pn"], fuse_batch=True,
                    fused_activation=fa, mcast_in0=True)
            run(fam, lambda pc=pc: ttnn.linear(xd, wd, bias=bd, program_config=pc, compute_kernel_config=mck,
                                               dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG), p)
        # minimal_matmul over its blocks (no bias+act combination issue: it takes both)
        mact = {"fused_activation": (ttnn.UnaryOpType.GELU, False)} if act else {}
        for mb, kb, nb, sh, sw in itertools.product((8, 16), (2, 4, 8), (4, 8), (1, 2, 4), (2, 4)):
            if kt % kb or sh * sw > cap or mb % sh or nb % sw:
                continue
            mc = ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw,
                                          compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y))
            run("mm", lambda mc=mc: ttnn.experimental.minimal_matmul(
                xd, wd, bias_tensor=bd, compute_kernel_config=mck, dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, config=mc, **mact), dict(mb=mb, kb=kb, nb=nb, sh=sh, sw=sw))
        ttnn.deallocate(xd); ttnn.deallocate(wd); ttnn.deallocate(bd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
