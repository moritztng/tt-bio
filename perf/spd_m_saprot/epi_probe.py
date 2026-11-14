"""How much of a SaProt projection's minimal_matmul time is its epilogue (the per-output-tile unpack of the fp32
partials, bias add and pack): tt-bio's transcription (kernels/mm_gate) with bias, against the same program with the
epilogue compiled out (MM_EPI_PROBE, wrong output), at the page shape, per projection and block config. No GELU.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/epi_probe.py --model saprot-650m [--fast] --out probe.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn

MODELS = {"saprot-650m": (1280, 5120), "saprot-1.3b": (1280, 5120)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m", choices=sorted(MODELS))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--cfgs", default="fc1:8,8,8,1,4;fc1:8,4,8,1,4;fc1:4,8,8,1,4;fc1:8,8,4,1,4;fc2:8,8,8,1,2;qkv:8,8,8,1,4;"
                    "out:16,4,4,1,4", help="';'-separated shape:M,K,N,sh,sw")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T, mm_generic as MG
    from tt_bio.mm_gate import KERNEL_DIR
    from tt_bio import saprot as S
    T.set_fast_mode(a.fast)
    dev = T.get_device()
    cfg, _ = S.CONFIGS[a.model]
    ck = S.Saprot(**cfg).compute_kernel_config
    d, ff = MODELS[a.model]
    shapes = {"qkv": (d, 3 * d), "out": (d, d), "fc1": (d, ff), "fc2": (ff, d)}
    wdt = ttnn.bfloat8_b if a.fast else ttnn.bfloat16
    g = dev.compute_with_storage_grid_size()
    grid = (g.x, g.y)
    gen = torch.Generator().manual_seed(0)
    B, L = 7, 544
    out = open(a.out, "a")
    for spec in filter(None, a.cfgs.split(";")):
        name, c = spec.split(":")
        bc = tuple(int(v) for v in c.split(","))
        K, N = shapes[name]
        xd = ttnn.from_torch(torch.randn(B, L, K, generator=gen).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)
        wd = ttnn.from_torch((torch.randn(K, N, generator=gen) / math.sqrt(K)).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=wdt)
        bd = ttnn.from_torch((torch.randn(1, N, generator=gen) * 0.1).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16)
        lin = S._SaprotLin.__new__(S._SaprotLin)
        lin.compute_kernel_config, lin.norm_weight = ck, torch.empty(d)
        mck = lin._ck(wd)
        o = ttnn.allocate_tensor_on_device(ttnn.Shape([B, L, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                           ttnn.DRAM_MEMORY_CONFIG)
        rec = dict(model=a.model, fast=a.fast, shape=name, cfg=bc, fidelity=str(mck.math_fidelity))
        for arm, probe in (("full", False), ("noepi", True)):
            try:
                e = MG.build(dev, xd, wd, [o], (bc, grid), MG.ckc_args(mck), kernel_dir=KERNEL_DIR, bias=bd)
                comp = e["kernels"][4]
                comp.kernel_source = str(KERNEL_DIR / "compute.cpp")
                comp.defines = [("FUSE_BIAS", "1")] + ([("MM_EPI_PROBE", "1")] if probe else [])
                pd = ttnn.ProgramDescriptor(kernels=e["kernels"], semaphores=e["semaphores"], cbs=e["cbs"])
                run = lambda: ttnn.generic_op([xd, wd, o], pd)
                for _ in range(2):
                    run()
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(a.calls):
                    run()
                ttnn.synchronize_device(dev)
                rec[f"{arm}_ms"] = round((time.perf_counter() - t0) / a.calls * 1e3, 4)
            except RuntimeError as exc:
                rec[f"{arm}_error"] = str(exc).splitlines()[0][:200]
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
        for t in (xd, wd, bd, o):
            ttnn.deallocate(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
