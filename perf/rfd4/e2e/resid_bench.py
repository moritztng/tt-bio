"""The z residual write at RFD4's pair shape, arm by arm: synced wall per call and bits against the base.

    TT_VISIBLE_DEVICES=<card> python3 perf/rfd4/e2e/resid_bench.py [--reps 50]

base: add(z, typecast(u, f32)) then typecast(z_new, bf16), what trunk._residual + _as do per sublayer.
mixed: add(z, u) on the mixed fp32 + bf16 operands, then the same bf16 cast (wrong: rounds z to bf16).
dual: rne_add.residual_dual, both results from one kernel.
"""
import argparse
import statistics
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

from tt_bio.tenstorrent import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    dev = get_device()
    g = torch.Generator().manual_seed(0)
    n = a.n
    zt = torch.randn(1, n, n, 384, generator=g) * 30
    ut = torch.randn(1, n, n, 384, generator=g)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    z, u = up(zt, ttnn.float32), up(ut, ttnn.bfloat16)

    def base():
        uf = ttnn.typecast(u, ttnn.float32)
        out = ttnn.add(z, uf)
        ttnn.deallocate(uf)
        return out, ttnn.typecast(out, ttnn.bfloat16)

    def mixed():
        out = ttnn.add(z, u)
        return out, ttnn.typecast(out, ttnn.bfloat16)

    from tt_bio import rne_add

    def dual():
        return rne_add.residual_dual(z, u)

    arms = {"base": base, "mixed": mixed, "dual": dual}
    ref = None
    for name, f in arms.items():
        o, ob = f()
        of, obf = ttnn.to_torch(o), ttnn.to_torch(ob)
        if ref is None:
            ref = (of, obf)
        print(f"{name}: out dtype {o.dtype}, max|d| f32 {(of - ref[0]).abs().max().item():.3e} "
              f"bf16 {(obf.float() - ref[1].float()).abs().max().item():.3e} "
              f"vs f64 {(of.double() - (zt.double() + ut.bfloat16().double())).abs().max().item():.3e}")
        ttnn.deallocate(o)
        ttnn.deallocate(ob)
    for rnd in range(2):
        for name, f in arms.items():
            ts = []
            for _ in range(a.reps):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o, ob = f()
                ttnn.synchronize_device(dev)
                ts.append(time.perf_counter() - t0)
                ttnn.deallocate(o)
                ttnn.deallocate(ob)
            print(f"round {rnd} {name}: median {statistics.median(ts) * 1e6:.1f} us")


if __name__ == "__main__":
    main()
