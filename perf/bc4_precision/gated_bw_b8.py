#!/usr/bin/env python3
"""The fused gated backward (`reblock_permute_gated_bw`) on a bf16 and a bfp8 projection, graded
against float64 on the SAME operands (the bfp8 projection dequantised), unpacked and packed.

A correct kernel reads at the bf16 envelope on both dtypes; a wrong one reads O(0.1-1). Written
because the bfp8 form shipped stepping its bf16 dp/dg tiles by the bfp8 tile size.

    TT_VISIBLE_DEVICES=<card> PYTHONPATH=. python perf/bc4_precision/gated_bw_b8.py
"""
import json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import reblock_permute as R
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).double()  # noqa: E731
    rows = []
    for N, C in ((224, 128), (64, 128)):
        torch.manual_seed(0)
        xw = torch.randn(1, N, N, 4 * C) * 1.5
        da = (torch.randn(1, C, N, N) * 1e-2).bfloat16().float()
        dad = ttnn.from_torch(da, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
            xd = ttnn.from_torch(xw, layout=ttnn.TILE_LAYOUT, dtype=dt, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            assert R.eligible_gated_bw(dad, xd), (N, C, dt)
            x64, d64 = down(xd), da.double()
            for p_off, g_off in ((0, C), (2 * C, 3 * C)):
                s = torch.sigmoid(x64[..., g_off:g_off + C])
                dap = d64.permute(0, 2, 3, 1)
                rdp, rdg = dap * s, dap * x64[..., p_off:p_off + C] * s * (1 - s)
                dp, dg = R.reblock_permute_gated_bw(dad, xd, p_off, g_off)
                slab = ttnn.from_torch(torch.zeros(1, N, N, 4 * C), layout=ttnn.TILE_LAYOUT,
                                       dtype=ttnn.bfloat16, device=dev,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
                R.reblock_permute_gated_bw(dad, xd, p_off, g_off, out=slab)
                sl = down(slab)
                row = {"N": N, "C": C, "dtype": str(dt), "p_off": p_off,
                       "dp": rel(down(dp), rdp), "dg": rel(down(dg), rdg),
                       "packed_dp": rel(sl[..., p_off:p_off + C], rdp),
                       "packed_dg": rel(sl[..., g_off:g_off + C], rdg)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    worst = max(max(r["dp"], r["dg"], r["packed_dp"], r["packed_dg"]) for r in rows)
    print(f"worst rel L2 {worst:.4g}: {'PASS' if worst < 2e-2 else 'FAIL'}")
    sys.exit(0 if worst < 2e-2 else 1)


if __name__ == "__main__":
    main()
