"""Does a token-axis linear give design k the same bits at batch B as at batch 1?

The trunk's 768 -> 3072 transition linears do not (one bf16 ULP, independence.py): ttnn's automatic
matmul config is chosen per shape, and fp32 dest with packer L1 accumulation sums the K blocks in an
order that depends on the blocking. This tries ttnn's automatic config and pinned configs whose
K blocking does not depend on M, and reports per config: bit-exact vs batch 1, error against a
float64 reference, and device ms at both widths.
"""

import argparse
import json
import sys
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import ttnn

    from tt_bio import ops
    from tt_bio.rfd4.trunk import compute_kernel_config
    from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device

    dev = get_device()
    ckc = compute_kernel_config()
    gx, gy = CORE_GRID_MAIN.x, CORE_GRID_MAIN.y
    B, n = a.batch, a.n
    g = torch.Generator().manual_seed(0)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731

    def configs(M, K, N):
        Mt, Kt, Nt = M // 32, K // 32, N // 32
        out = {"auto": None}
        for w in (1, 2, 3, 4, 6, 8):
            if Kt % w:
                continue
            pn = -(-Nt // (gx * gy))
            sw = max(d for d in range(1, min(pn, 8) + 1) if pn % d == 0)
            out[f"1d_w{w}"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=w, out_subblock_h=1,
                out_subblock_w=sw, per_core_M=Mt, per_core_N=pn, fuse_batch=True,
                fused_activation=None, mcast_in0=True)
        return out

    rows = []
    for K, N in ((768, 3072), (3072, 768)):
        x = torch.randn(B, n, K, generator=g)
        w = torch.randn(K, N, generator=g) / K ** 0.5
        xb, wb = x.bfloat16(), w.bfloat16()
        ref = xb.double() @ wb.double()
        wt = up(w)
        for name in configs(n, K, N):
            res, ms = {}, {}
            for width in (1, B):
                cfg = configs(n * width, K, N)[name]
                kw = {} if cfg is None else {"program_config": cfg}
                xt = up(x[:width])
                try:
                    f = (lambda: ops.linear(xt, wt, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                            core_grid=CORE_GRID_MAIN)) if cfg is None else (
                        lambda: ttnn.linear(xt, wt, compute_kernel_config=ckc, dtype=ttnn.bfloat16, **kw))
                    y = f()
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(20):
                        ttnn.deallocate(y)
                        y = f()
                    ttnn.synchronize_device(dev)
                    ms[width] = (time.perf_counter() - t0) / 20 * 1e3
                    res[width] = ttnn.to_torch(y).float()
                    ttnn.deallocate(y)
                except RuntimeError as exc:
                    res[width] = str(exc).splitlines()[0][:160]
                ttnn.deallocate(xt)
            row = {"K": K, "N": N, "config": name}
            if all(isinstance(r, torch.Tensor) for r in res.values()):
                b1 = res[1][0]
                row["bit_exact_vs_b1"] = bool(torch.equal(res[B][0], b1))
                row["rel_err_vs_f64"] = float((res[B].double() - ref).abs().max() / ref.abs().max())
                row["ms_b1"], row[f"ms_b{B}"] = round(ms[1], 4), round(ms[B], 4)
            else:
                row["error"] = {k: v for k, v in res.items() if isinstance(v, str)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        ttnn.deallocate(wt)
    with open(a.out, "w") as f:
        json.dump({"batch": B, "n": n, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
