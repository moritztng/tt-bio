"""Is the shipped OpenDDE _MM_BLOCK entry numerically sound on the 8x9 Wormhole grid?

On Blackhole's 110 cores these two entries are not bit-exact but differ by exactly one bf16 ULP
(max_abs 0.5 on the qkv, 0.25 on the gate) with the relative RMSD against an fp32 reference
unchanged to four significant figures -- that is the evidence the entries shipped on.
The 8x9 probe read max_abs 128, which is ~128 ULPs, so either the peak magnitude assumption is
wrong or the configured op is producing wrong values on this grid. This decides which.
"""
import os, json
import torch
import ttnn

out = {"arch": ttnn.get_arch_name()}
dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()
    GX, GY = int(g.x), int(g.y)
    out["grid"] = [GX, GY]
    ckc = ttnn.types.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    res = {}
    for S, N, entry in ((320, 384, (8, 12, 1, 2, 1)), (320, 1152, (4, 12, 1, 2, 1))):
        xt = torch.randn(1, S, S, 384, dtype=torch.bfloat16)
        wt = torch.randn(384, N, dtype=torch.bfloat16)
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        M, K, Nb, sh, sw = entry
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=M, K_block_size=K, N_block_size=Nb, subblock_h=sh, subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY))
        r0 = ttnn.to_torch(ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=None)).float()
        r1 = ttnn.to_torch(ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg)).float()
        ref = (xt.float().reshape(-1, 384) @ wt.float()).reshape(r0.shape)
        d = (r0 - r1).abs()
        peak = float(ref.abs().max())
        # bf16 ULP at the peak magnitude
        import math
        ulp = 2.0 ** (math.floor(math.log2(peak)) - 7)
        res["S%d_384x%d" % (S, N)] = {
            "entry": list(entry),
            "peak_abs_ref": peak, "bf16_ulp_at_peak": ulp,
            "max_abs_base_vs_cfg": float(d.max()),
            "max_abs_in_ulps": float(d.max()) / ulp,
            "n_elems": int(d.numel()),
            "n_diff_gt_ulp": int((d > ulp).sum()),
            "frac_diff_gt_ulp": float((d > ulp).float().mean()),
            "relRMSD_base_vs_fp32": float((r0 - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
            "relRMSD_cfg_vs_fp32": float((r1 - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
            "max_abs_base_vs_fp32": float((r0 - ref).abs().max()),
            "max_abs_cfg_vs_fp32": float((r1 - ref).abs().max()),
        }
        ttnn.deallocate(x); ttnn.deallocate(w)
finally:
    ttnn.close_device(dev)
out["results"] = res
print("MM12_JSON " + json.dumps(out, indent=2))
