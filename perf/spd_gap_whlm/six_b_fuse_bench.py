"""ESMC-6B's projections at the page shape (one 544-row sequence, bf16 weights = normal mode): what it runs today
(ttnn.linear on the LIN6B program, then a 3-way chunk copy for qkv and a separate residual add for out_proj and fc2)
against minimal_matmul doing the split or the residual add itself (minimal_matmul_split,
dit_minimal_matmul_addcmul_fused). Per arm: median device ms and error vs float64.

    python3 perf/spd_gap_whlm/six_b_fuse_bench.py --out six_b_fuse.jsonl
"""
import argparse
import json

import torch
import ttnn

from qkfuse_bench import timed

D, FF, ROWS = 2560, 6912, 544


def err(got, ref):
    e = got - ref
    return {"rel_l2": float(e.norm() / ref.norm()), "max_abs": float(e.abs().max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio import esmc
    dev = T.get_device()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    g = torch.Generator().manual_seed(0)
    bf = lambda t: t.to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(bf(t), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    x = bf(torch.randn(1, ROWS, D, generator=g))
    h = bf(torch.randn(1, ROWS, FF, generator=g))
    r = bf(4 * torch.randn(1, ROWS, D, generator=g))
    wqkv = bf(torch.randn(D, 3 * D, generator=g) * D ** -0.5)
    wout = bf(torch.randn(D, D, generator=g) * D ** -0.5)
    wfc2 = bf(torch.randn(FF, D, generator=g) * FF ** -0.5)
    xd, hd, rd = up(x), up(h), up(r)
    wq, wo, wf = up(wqkv), up(wout), up(wfc2)
    ones = ttnn.from_torch(torch.ones(1, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def lin(a, w):
        pc = esmc._lin6b_program(a, w)
        return ttnn.linear(a, w, compute_kernel_config=ck, **({"program_config": pc} if pc is not None else {}))

    def mmcfg(a, w):
        return esmc._mm_config.__wrapped__(a, w) if hasattr(esmc._mm_config, "__wrapped__") else esmc._mm_config(a, w)

    def qkv_chunk():
        o = lin(xd, wq)
        parts = ttnn.chunk(o, 3, dim=-1)
        ttnn.deallocate(o)
        return list(parts)

    def qkv_split():
        return list(ttnn.experimental.minimal_matmul_split(
            xd, wq, chunks=3, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **mmcfg(xd, wq)))

    def resid_add(a, w):
        def f():
            o = lin(a, w)
            s = ttnn.add(rd, o)
            ttnn.deallocate(o)
            return [s]
        return f

    def resid_fused(a, w):
        def f():
            return [ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                a, w, 1.0, rd, ones, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, **mmcfg(a, w))]
        return f

    q64 = x.double() @ wqkv.double()
    refs = {"qkv": [q64[..., i * D:(i + 1) * D] for i in range(3)],
            "out": [r.double() + x.double() @ wout.double()],
            "fc2": [r.double() + h.double() @ wfc2.double()]}
    sites = {"qkv": (qkv_chunk, qkv_split), "out": (resid_add(xd, wo), resid_fused(xd, wo)),
             "fc2": (resid_add(hd, wf), resid_fused(hd, wf))}
    out = open(args.out, "w")
    for site, (base, new) in sites.items():
        rec = {"site": site}
        for arm, fn in (("today", base), ("fused", new)):
            try:
                o = fn()
                rec[arm] = [err(ttnn.to_torch(t).double().reshape(refs[site][i].shape), refs[site][i])
                            for i, t in enumerate(o)]
                for t in o:
                    ttnn.deallocate(t)
            except Exception as exc:  # noqa: BLE001  a refused program is a result too
                rec[arm] = {"error": str(exc)[:300]}
        ms = {"today": [], "fused": []}
        for _ in range(args.repeat):
            for arm, fn in (("today", base), ("fused", new)):
                if isinstance(rec[arm], dict):
                    continue
                ms[arm].append(timed(lambda fn=fn: [ttnn.deallocate(t) for t in fn()], dev, 1, args.inner))
        for arm, ts in ms.items():
            if ts:
                rec[arm + "_ms"] = sorted(ts)[len(ts) // 2]
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
