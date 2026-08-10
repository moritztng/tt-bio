#!/usr/bin/env python3
"""Is the L1 output bit-exact against PRODUCTION TODAY at the same in0_block_w?

The release-gated arm keeps main's `in0_block_w`=8 and only moves the destination. If a memory
config cannot change a value, this must be `torch.equal` against the DRAM output of the identical
config -- which is a stronger and more useful claim than being bit-exact against the untuned
`core_grid` reference, because main is not bit-exact against that either.
"""
import json, sys
from pathlib import Path
import torch, ttnn
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.tenstorrent as T

TOK, C_Z = 298, 256
dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
res = {"card": "qb2 chip 2, ttnn 0.68.0", "grid": list(T.COMPUTE_GRID_MAIN), "shape": [1, TOK, TOK, C_Z]}

x = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
wz = ttnn.from_torch(torch.randn(C_Z, C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
wg = ttnn.from_torch(torch.randn(C_Z, C_Z) * 0.05, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
z0 = torch.randn(1, TOK, TOK, C_Z) * 0.1


def cmp(name, ref, got):
    eq = bool(torch.equal(ref, got))
    d = (ref.double() - got.double()).abs().max().item()
    res[name] = {"torch_equal": eq, "max_abs": d}
    print(f"  {name:40s} torch.equal={eq}  max_abs={d:.6g}", flush=True)


T._PAIR_PROJ_BW, T._PAIR_PROJ_L1_BW = 16, 16
T._PAIR_PROJ_L1_OUT = False
T._pair_proj_program_config.cache_clear()
ref = ttnn.to_torch(T._pair_proj_linear(x, wz, ckc, ttnn.bfloat16))         # production today, DRAM
T._PAIR_PROJ_L1_OUT = True
got = ttnn.to_torch(T._pair_proj_linear(x, wz, ckc, ttnn.bfloat16, l1_out=True))
cmp("pair_proj_L1out_bw8_vs_production_today", ref, got)


def chain(l1_out):
    z = ttnn.from_torch(z0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    p = T._pair_proj_linear(x, wz, ckc, ttnn.bfloat16, l1_out=l1_out)
    g = T._pair_proj_linear(x, wg, ckc, ttnn.bfloat16, l1_out=l1_out)
    o = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(g)
    ttnn.add_(z, o)
    ttnn.deallocate(o)
    out = ttnn.to_torch(z)
    ttnn.deallocate(z)
    return out


T._PAIR_PROJ_L1_OUT = False
ref_chain = chain(False)
T._PAIR_PROJ_L1_OUT = True
cmp("trimul_chain_L1out_bw8_vs_production_today", ref_chain, chain(True))

Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print("wrote", sys.argv[1], flush=True)
