"""Kill gate 1: _fp32_softmax_attention sharded vs interleaved, at the production shapes."""
import json, sys
import torch, ttnn
sys.path.insert(0, ".")
import tt_bio.tenstorrent as T

dev = T.get_device()
DRAM = ttnn.DRAM_MEMORY_CONFIG
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
res = {}
# the two OpenFold3 site classes the census reports, plus the 768/1024 leading dims
CASES = [("512_hd128", 512, 4, 512, 128), ("512_hd64", 512, 4, 512, 64),
         ("256_hd128", 256, 4, 256, 128), ("768_hd128", 768, 4, 768, 128)]
for name, rows, h, s, hd in CASES:
    try:
        torch.manual_seed(0)
        q = ttnn.from_torch(torch.randn(rows, h, s, hd) * 0.3, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        k = ttnn.from_torch(torch.randn(rows, h, s, hd) * 0.3, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        v = ttnn.from_torch(torch.randn(rows, h, s, hd) * 0.3, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        b = ttnn.from_torch(torch.randn(1, h, s, s) * 0.5, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        out = {}
        for arm, budget in (("noshard", 0), ("on", 768 << 10)):
            T._FP32_SOFTMAX_L1_BYTES_PER_CORE = budget
            T.FP32_SOFTMAX_STATS.update(calls=0, blocked=0, blocks=0, fused=0, unfused=0,
                                        l1=0, l1_blocks=0)
            o = T._fp32_softmax_attention(q, k, v, b, hd ** -0.5, ckc)
            out[arm] = ttnn.to_torch(o)
            ttnn.deallocate(o)
            res[f"{name}_{arm}_stats"] = dict(T.FP32_SOFTMAX_STATS)
        d = (out["noshard"].float() - out["on"].float()).abs().max().item()
        res[f"{name}_equal"] = bool(torch.equal(out["noshard"], out["on"]))
        res[f"{name}_maxabs"] = d
        for t in (q, k, v, b):
            ttnn.deallocate(t)
    except Exception as e:
        res[f"{name}_error"] = f"{type(e).__name__}: {e}"[:260]
T._FP32_SOFTMAX_L1_BYTES_PER_CORE = 768 << 10
print("RESULT " + json.dumps(res, indent=1))
