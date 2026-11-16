"""Device check of mm_resid's row statistics (the producer half of an in-matmul LayerNorm) at ESMC's residual shapes.

Per shape and fidelity, three arms of resid + x @ w on the page batch (7 x 544 rows):
  wheel  ttnn.experimental.dit_minimal_matmul_addcmul_fused, what ESMC runs today
  resid  tt_bio.mm_resid.matmul_add, same blocking (esmc._mm_config), zero bias
  stats  the same with rowstats=True
Reports median ms per call (synchronised loops, so host dispatch is included alike in every arm), whether `stats`
writes the same bytes as `resid`, max |resid - wheel|, and the statistics against float64 sums of the bf16 output
the arm wrote (sum x, sum of bf16(x^2)): max relative error of each, and of the rstd they give.

    python perf/spd_m_esmc/rowstats_check.py --out rowstats.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spd_gap_whlm"))
from qkfuse_bench import timed  # noqa: E402

ROWS = 7 * 544
# name: (K, N) of the two residual producers, out_proj (d -> d) and fc2 (d_ff -> d)
SHAPES = {"300m-out": (960, 960), "300m-fc2": (2560, 960), "600m-out": (1152, 1152), "600m-fc2": (3072, 1152)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--fidelities", default="HiFi2,HiFi4,LoFi")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio import esmc, mm_resid
    dev = T.get_device()
    out = open(a.out, "w")
    for name in a.shapes.split(","):
        k, n = SHAPES[name]
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(ROWS, k, generator=gen).to(torch.bfloat16)
        w = (torch.randn(k, n, generator=gen) * k ** -0.5).to(torch.bfloat16)
        # a residual stream with ESMC's shape of trouble: one outlier channel far above the rest
        r = torch.randn(ROWS, n, generator=gen)
        r[:, 7] *= 40.0
        r = r.to(torch.bfloat16)
        xd, wd, rd = (ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16) for t in (x, w, r))
        zero = ttnn.from_torch(torch.zeros(1, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        ones = ttnn.from_torch(torch.ones(1, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        c = esmc._mm_config(xd, wd)["config"]
        g = c.compute_with_storage_grid_size
        cfg, grid = (c.M_block_size, c.K_block_size, c.N_block_size, c.subblock_h, c.subblock_w), (int(g.x), int(g.y))
        for fid in a.fidelities.split(","):
            ck = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)
            wheel = lambda: ttnn.experimental.dit_minimal_matmul_addcmul_fused(  # noqa: E731
                xd, wd, 1.0, rd, ones, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, config=c)
            resid = lambda: mm_resid.matmul_add(xd, wd, zero, rd, cfg, grid, ck)  # noqa: E731
            stats = lambda: mm_resid.matmul_add(xd, wd, zero, rd, cfg, grid, ck, rowstats=True)  # noqa: E731
            row = {"shape": name, "fidelity": fid, "cfg": cfg, "grid": grid}
            for arm, fn in (("wheel", wheel), ("resid", resid), ("stats", stats)):
                def run(fn=fn):
                    o = fn()
                    for t in (o if isinstance(o, tuple) else (o,)):
                        ttnn.deallocate(t)
                row[arm + "_ms"] = timed(run, dev, a.repeat, a.inner)
            ow, orr = ttnn.to_torch(wheel()), ttnn.to_torch(resid())
            os_, st = stats()
            os_, st = ttnn.to_torch(os_), ttnn.to_torch(st).double()
            row["stats_out_identical"] = bool(torch.equal(os_, orr))
            row["resid_vs_wheel_maxabs"] = (orr.float() - ow.float()).abs().max().item()
            xb = os_.reshape(ROWS, n).double()
            s1 = xb.sum(-1)
            s2 = (xb * xb).to(torch.bfloat16).double().sum(-1)
            nb = st.shape[-1] // 64
            t = st.reshape(ROWS, nb, 2, 32)
            row["stats_cols_uniform"] = bool((t == t[..., :1]).all())
            d1, d2 = t[..., 0, 0].sum(-1), t[..., 1, 0].sum(-1)
            row["sum_x_maxrel"] = ((d1 - s1).abs() / xb.abs().sum(-1)).max().item()
            row["sum_x2_maxrel"] = ((d2 - s2).abs() / s2).max().item()
            rs = lambda a1, a2: torch.rsqrt(a2 / n - (a1 / n) ** 2 + 1e-5)  # noqa: E731
            xc = xb - xb.mean(-1, keepdim=True)
            r64 = torch.rsqrt((xc * xc).mean(-1) + 1e-5)
            row["rstd_maxrel_vs_exact_pc"] = (rs(d1, d2) / rs(s1, s2) - 1).abs().max().item()
            row["rstd_maxrel_vs_twopass_f64"] = (rs(d1, d2) / r64 - 1).abs().max().item()
            print(json.dumps(row), flush=True)
            out.write(json.dumps(row) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
