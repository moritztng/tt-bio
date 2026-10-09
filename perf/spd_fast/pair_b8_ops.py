"""What a bfp8 pair stream would buy in fast mode: ms per call of the bandwidth-bound pair ops, bf16 vs bfp8.

    TT_VISIBLE_DEVICES=N python perf/spd_fast/pair_b8_ops.py --out OUT.json [--S 736]

Each row times one op on the whole [1, S, S, 256] pair tensor as the pairformer calls it (layer norm, residual add,
narrow projection, gate multiply, pair transpose), once with bf16 operands and once with bfp8 where the op takes it,
and quotes the relative error of the bfp8 result against the bf16 one. Fast mode's census (state/spd/CENSUS.md) puts
~115 s of the trunk in these ops at c730; this says which dtype changes are worth wiring before any fold is spent.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WARM, REPS = 2, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--S", type=int, default=736)
    a = ap.parse_args()
    import torch, ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    dev = T.get_device()
    S, C = a.S, 256
    BF, B8 = ttnn.bfloat16, ttnn.bfloat8_b
    DR = ttnn.DRAM_MEMORY_CONFIG
    ckc = (ttnn.WormholeComputeKernelConfig if T.is_wormhole() else ttnn.types.BlackholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True)
    g = torch.Generator().manual_seed(0)
    zt = torch.randn(1, S, S, C, generator=g)
    ut = 0.1 * torch.randn(1, S, S, C, generator=g)
    w = torch.randn(C, C, generator=g) / C ** 0.5
    lw, lb = 1 + 0.1 * torch.randn(C, generator=g), 0.1 * torch.randn(C, generator=g)
    up = lambda t, dt: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt, memory_config=DR)
    z = {BF: up(zt, BF), B8: up(zt, B8)}
    u = {BF: up(ut, BF), B8: up(ut, B8)}
    wt = {BF: up(w, BF), B8: up(w, B8)}
    lwt = ttnn.from_torch(lw.reshape(1, C), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=BF)
    lbt = ttnn.from_torch(lb.reshape(1, C), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=BF)
    res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"),
           "arch": "wormhole" if T.is_wormhole() else "blackhole", "S": S, "rows": []}

    def timed(name, fn):
        try:
            for _ in range(WARM):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            ts = []
            for _ in range(REPS):
                t0 = time.perf_counter(); o = fn(); ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(o)
            o = fn()
            r = {"op": name, "ms": round(st.median(ts), 3), "spread": round(max(ts) - min(ts), 3),
                 "dtype": str(o.dtype)}
            r["_out"] = ttnn.to_torch(o).float()
            ttnn.deallocate(o)
        except Exception as e:  # an op that refuses bfp8 is a result, not a crash
            r = {"op": name, "err": str(e).splitlines()[0][:200]}
        print(json.dumps({k: v for k, v in r.items() if k != "_out"}), flush=True)
        res["rows"].append(r)
        return r

    ln = lambda x, dt: lambda: ttnn.layer_norm(x, weight=lwt, bias=lbt, epsilon=1e-5, memory_config=DR,
                                               compute_kernel_config=ckc, **({} if dt is None else {"dtype": dt}))
    cases = [
        ("ln bf16->bf16", ln(z[BF], None)), ("ln bf16->bfp8", ln(z[BF], B8)), ("ln bfp8->bfp8", ln(z[B8], B8)),
        ("add bf16+bf16", lambda: ttnn.add(z[BF], u[BF], memory_config=DR)),
        ("add bf16+bfp8->bf16", lambda: ttnn.add(z[BF], u[B8], memory_config=DR, dtype=BF)),
        ("add bfp8+bfp8->bfp8", lambda: ttnn.add(z[B8], u[B8], memory_config=DR)),
        ("mul bf16*bf16", lambda: ttnn.multiply(z[BF], u[BF], memory_config=DR)),
        ("mul bfp8*bfp8->bfp8", lambda: ttnn.multiply(z[B8], u[B8], memory_config=DR)),
        ("linear bf16 x bf16 256->256", lambda: ttnn.linear(z[BF], wt[BF], memory_config=DR, compute_kernel_config=ckc)),
        ("linear bfp8 x bfp8 256->256 ->bfp8", lambda: ttnn.linear(z[B8], wt[B8], memory_config=DR,
                                                                    compute_kernel_config=ckc, dtype=B8)),
        ("linear bf16 x bfp8 256->256 ->bfp8", lambda: ttnn.linear(z[BF], wt[B8], memory_config=DR,
                                                                    compute_kernel_config=ckc, dtype=B8)),
        ("permute(0,2,1,3) bf16", lambda: ttnn.permute(z[BF], (0, 2, 1, 3), memory_config=DR)),
        ("permute(0,2,1,3) bfp8", lambda: ttnn.permute(z[B8], (0, 2, 1, 3), memory_config=DR)),
    ]
    out = {}
    for name, fn in cases:
        out[name] = timed(name, fn)
    for b, q in [("ln bf16->bf16", "ln bf16->bfp8"), ("ln bf16->bf16", "ln bfp8->bfp8"),
                 ("add bf16+bf16", "add bf16+bfp8->bf16"), ("add bf16+bf16", "add bfp8+bfp8->bfp8"),
                 ("mul bf16*bf16", "mul bfp8*bfp8->bfp8"),
                 ("linear bf16 x bf16 256->256", "linear bfp8 x bfp8 256->256 ->bfp8"),
                 ("linear bf16 x bf16 256->256", "linear bf16 x bfp8 256->256 ->bfp8"),
                 ("permute(0,2,1,3) bf16", "permute(0,2,1,3) bfp8")]:
        if "_out" in out[b] and "_out" in out[q]:
            ref, x = out[b]["_out"], out[q]["_out"]
            out[q]["rel_err_vs_bf16"] = float((x - ref).norm() / ref.norm())
            out[q]["speedup"] = round(out[b]["ms"] / out[q]["ms"], 3)
    for r in res["rows"]:
        r.pop("_out", None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["rows"], indent=1))


if __name__ == "__main__":
    main()
