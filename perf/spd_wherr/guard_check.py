"""tt_bio.dest_guard on the shapes where the erratum was measured: wrong pixels and op time, guard on and off.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_wherr/guard_check.py --out guard.json

Opens the device through tt_bio (which installs the guard on Wormhole), then for each case runs the call as tt_bio
writes it, with the guard on and with it bypassed, against float64 (site_probe.py's rule and inputs, draw 0).
"""
import argparse, json, statistics, time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

from tt_bio import dest_guard, tenstorrent as TT  # noqa: E402

CASES = [  # name, m, k, n, batch (leading dim of a 3D input; 1 = 2D), call
    ("dit_qkv_auto", 2560, 768, 3072, 5, "linear"),
    ("dit_o_auto", 2560, 768, 768, 5, "linear"),
    ("mm_default", 262144, 256, 768, 1, "minimal_matmul"),
    ("mm_triatt_K8", 262144, 256, 256, 1, "minimal_matmul_k8"),
    ("matmul_ibw4", 8192, 768, 768, 1, "matmul_ibw4"),
]


def wrong(R, Y):
    err = (Y - R).abs()
    ulp = torch.exp2(torch.floor(torch.log2(R.abs().clamp_min(1e-30))) - 7)
    return int(((err > 8 * ulp) & (err > 16 * err.pow(2).mean().sqrt()) & (err > 0.25)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = TT.get_device()
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"arch": str(dev.arch()), "guard_installed": dest_guard.active(), "cases": []}
    for name, m, k, n, batch, call in CASES:
        torch.manual_seed(0)
        A = torch.randn(m, k).bfloat16(); W = (torch.randn(k, n) / k ** 0.5).bfloat16()
        R = A.double() @ W.double()
        ta = ttnn.from_torch(A.reshape(batch, m // batch, k) if batch > 1 else A, layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16)
        tw = ttnn.from_torch(W, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        if call == "linear":
            run = lambda: ttnn.linear(ta, tw, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                      core_grid=ttnn.CoreGrid(y=g.y, x=g.x))
        elif call.startswith("minimal_matmul"):
            cfg = None if call == "minimal_matmul" else ttnn.MinimalMatmulConfig(
                M_block_size=4, K_block_size=8, N_block_size=1, subblock_h=4, subblock_w=1,
                compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y))
            run = lambda: ttnn.experimental.minimal_matmul(input_tensor=ta, weight_tensor=tw, compute_kernel_config=ck,
                                                           dtype=ttnn.bfloat16, config=cfg)
        else:
            mt, nt = m // 32, n // 32
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(g.x, g.y), in0_block_w=4, out_subblock_h=1,
                out_subblock_w=max(s for s in (1, 2, 3, 4) if (nt // g.x) % s == 0),
                out_block_h=mt // g.y, out_block_w=nt // g.x, per_core_M=mt // g.y, per_core_N=nt // g.x,
                transpose_mcast=False, fused_activation=None, fuse_batch=True)
            run = lambda: ttnn.matmul(ta, tw, compute_kernel_config=ck, dtype=ttnn.bfloat16, program_config=pc)
        for guard in (True, False):
            dest_guard._ON[0] = guard and res["guard_installed"]
            before = dict(dest_guard.STATS)
            y = run(); Y = ttnn.to_torch(y).double().reshape(m, n); ttnn.deallocate(y)
            ts = []
            for _ in range(5):
                ttnn.synchronize_device(dev); t0 = time.perf_counter(); y = run(); ttnn.synchronize_device(dev)
                ts.append(time.perf_counter() - t0); ttnn.deallocate(y)
            cell = dict(case=name, guard=guard, wrong=wrong(R, Y), el=m * n, us=round(statistics.median(ts) * 1e6, 1),
                        stats={k2: dest_guard.STATS[k2] - before[k2] for k2 in before})
            res["cases"].append(cell); print(json.dumps(cell), flush=True)
        dest_guard._ON[0] = res["guard_installed"]
        ttnn.deallocate(ta); ttnn.deallocate(tw)
    res["refused"] = {str(k): v for k, v in dest_guard._REFUSED.items()}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print("refused:", res["refused"])


if __name__ == "__main__":
    main()
