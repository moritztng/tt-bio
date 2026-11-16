"""Device check of the in-matmul LayerNorm at ESMC's FFN site: x = resid + h @ Wo through mm_resid with row statistics,
then silu(LN(x) W1) * LN(x) W2 by mm_gate reading the raw x (W'' = C diag(gamma) W, rstd and beta W in its epilogue),
against today's ttnn.layer_norm + mm_gate. Page batch rows (7 x 544), random weights at ESMC scale, one outlier channel.

Per shape and mode: median ms of today's LN + gate and of the LN-in-matmul gate (the producer's statistics cost is
rowstats_check.py's stats_ms - resid_ms), and rel_l2 of both against float64 of the bf16 x each read.

    python perf/spd_m_esmc/lngate_check.py --out lngate.jsonl
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
SHAPES = {"300m": (960, 2560), "600m": (1152, 3072)}   # d, d_ff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--modes", default="normal,fast")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tt_bio import metal_overlay
    metal_overlay.enable(("silu_f32",))
    from tt_bio import tenstorrent as T
    from tt_bio import esmc, mm_gate, mm_resid
    dev = T.get_device()
    gx = dev.compute_with_storage_grid_size()
    grid = (int(gx.x), int(gx.y))
    out = open(a.out, "w")
    for name in a.shapes.split(","):
        d, ff = SHAPES[name]
        gen = torch.Generator().manual_seed(0)
        h = torch.randn(ROWS, d, generator=gen).to(torch.bfloat16)
        wo = (torch.randn(d, d, generator=gen) * d ** -0.5).to(torch.bfloat16)
        r = torch.randn(ROWS, d, generator=gen)
        r[:, 7] *= 40.0
        r = r.to(torch.bfloat16)
        gamma = 1 + 0.2 * torch.randn(d, generator=gen)
        gamma[7] = 3.0
        beta = 0.1 * torch.randn(d, generator=gen)
        w1 = torch.randn(d, ff, generator=gen) * d ** -0.5
        w2 = torch.randn(d, ff, generator=gen) * d ** -0.5
        hd, wod, rd = (ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16) for t in (h, wo, r))
        zero = ttnn.from_torch(torch.zeros(1, d), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        c = esmc._mm_config(hd, wod)["config"]
        pcfg = (c.M_block_size, c.K_block_size, c.N_block_size, c.subblock_h, c.subblock_w)
        pck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                     math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        xd, st = mm_resid.matmul_add(hd, wod, zero, rd, pcfg, grid, pck, rowstats=True)
        xd3 = ttnn.reshape(xd, [7, 544, d])
        x64 = ttnn.to_torch(xd).double().reshape(ROWS, d)
        m = x64.mean(-1, keepdim=True)
        xn = (x64 - m) / torch.sqrt(((x64 - m) ** 2).mean(-1, keepdim=True) + 1e-5) * gamma.double() + beta.double()
        ref = torch.nn.functional.silu(xn @ w1.double()) * (xn @ w2.double())
        for mode in a.modes.split(","):
            fast = mode == "fast"
            wdt = ttnn.bfloat8_b if fast else ttnn.bfloat16
            bf16_acc = fast
            ck = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2 if fast else ttnn.MathFidelity.HiFi4,
                math_approx_mode=False, fp32_dest_acc_en=not bf16_acc, packer_l1_acc=True)
            cfg = mm_gate.config(xd3, 2 * ff, d // 32, grid, bf16_acc)
            group = cfg[2] // 2 * 32
            # today: LayerNorm then the gate over gamma/beta-free weights (the LN carries the affine)
            w12 = ttnn.from_torch(mm_gate.interleave(w1, w2, group), dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev)
            gd = ttnn.from_torch(gamma[None], layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            bd = ttnn.from_torch(beta[None], layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            lnck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                          math_approx_mode=False, fp32_dest_acc_en=True)

            def today():
                xn_ = ttnn.layer_norm(xd3, weight=gd, bias=bd, epsilon=1e-5, compute_kernel_config=lnck,
                                      memory_config=ttnn.L1_MEMORY_CONFIG)
                o = mm_gate.gated(xn_, w12, cfg, ck, grid, pack_sfpu=True)
                ttnn.deallocate(xn_)
                return o
            w1c, b1 = mm_gate.ln_weights(gamma, beta, w1)
            w2c, b2 = mm_gate.ln_weights(gamma, beta, w2)
            w12c = ttnn.from_torch(mm_gate.interleave(w1c.float(), w2c.float(), group), dtype=wdt,
                                   layout=ttnn.TILE_LAYOUT, device=dev)
            b12 = ttnn.from_torch(mm_gate.interleave(b1.float(), b2.float(), group), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
            lnmm = lambda: mm_gate.gated(xd3, w12c, cfg, ck, grid, pack_sfpu=True, ln=(st, b12))  # noqa: E731
            row = {"shape": name, "mode": mode, "cfg": cfg, "producer_cfg": pcfg, "stats_blocks": int(st.shape[-1]) // 64}
            for arm, fn in (("today", today), ("lnmm", lnmm)):
                o = fn()
                got = ttnn.to_torch(o).double().reshape(ROWS, ff)
                ttnn.deallocate(o)
                row[arm + "_rel_l2"] = ((got - ref).norm() / ref.norm()).item()
                row[arm + "_finite"] = bool(torch.isfinite(got).all())
                row[arm + "_ms"] = timed(lambda fn=fn: ttnn.deallocate(fn()), dev, a.repeat, a.inner)
            print(json.dumps(row), flush=True)
            out.write(json.dumps(row) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
