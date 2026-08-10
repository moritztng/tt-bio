#!/usr/bin/env python3
"""p3-l1-output — parity at the fold's own shapes, through the PRODUCTION helpers.

Not a re-implementation of the config: this calls `_pair_proj_linear` / `_narrow_proj_linear`
exactly as `_trimul_out_proj`, `gate_and_project` and `AttentionPairBias` call them, and compares
against the untuned `ttnn.linear(core_grid=)` reference at [1, 298, 320, 256].
"""
import json, sys
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tt_bio.tenstorrent as T  # noqa: E402

TOK, C_Z, N_HEADS = 298, 256, 16
dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
res = {"card": "qb1 card 1", "grid": list(T.COMPUTE_GRID_MAIN), "shape": [1, TOK, TOK, C_Z]}


def cmp(name, ref_t, got_t):
    r, g = ref_t.double(), got_t.double()
    d = (r - g).abs()
    eq = bool(torch.equal(ref_t, got_t))
    rr, gg = r.flatten(), g.flatten()
    pcc = float(torch.corrcoef(torch.stack([rr, gg]))[0, 1])
    res[name] = {"torch_equal": eq, "max_abs": float(d.max()),
                 "rmsd": float(((r - g) ** 2).mean().sqrt()), "pcc": round(pcc, 10)}
    print(f"  {name:34s} torch.equal={eq}  max_abs={float(d.max()):.6g}  pcc={pcc:.10f}", flush=True)


x = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
wz = ttnn.from_torch(torch.randn(C_Z, C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
wb = ttnn.from_torch(torch.randn(C_Z, N_HEADS), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
nw = ttnn.from_torch(torch.randn(C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
nb = ttnn.from_torch(torch.randn(C_Z) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev)

print("=== pair-track output projection, [1,298,320,256] @ [256,256] ===", flush=True)
ref = ttnn.to_torch(ttnn.linear(x, wz, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                                core_grid=T.CORE_GRID_MAIN))
T._PAIR_PROJ_L1_OUT = False
cmp("pair_proj_production_today_dram", ref,
    ttnn.to_torch(T._pair_proj_linear(x, wz, ckc, ttnn.bfloat16)))
T._PAIR_PROJ_L1_OUT, T._PAIR_PROJ_L1_BW = True, 1
cmp("pair_proj_l1_out_bw1", ref,
    ttnn.to_torch(T._pair_proj_linear(x, wz, ckc, ttnn.bfloat16, l1_out=True)))
T._PAIR_PROJ_L1_BW = 16
T._pair_proj_program_config.cache_clear()
cmp("pair_proj_l1_out_bw16cap", ref,
    ttnn.to_torch(T._pair_proj_linear(x, wz, ckc, ttnn.bfloat16, l1_out=True)))
T._PAIR_PROJ_L1_BW = 1
T._pair_proj_program_config.cache_clear()

print("=== AttentionPairBias z->bias, [1,298,320,256] @ [256,16] ===", flush=True)
zn_d = ttnn.layer_norm(x, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=ckc,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
zn_l = ttnn.layer_norm(x, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=ckc,
                       memory_config=ttnn.L1_MEMORY_CONFIG)
cmp("layer_norm_l1_vs_dram", ttnn.to_torch(zn_d), ttnn.to_torch(zn_l))
ref_b = ttnn.to_torch(ttnn.linear(zn_d, wb, compute_kernel_config=ckc,
                                  core_grid=T.CORE_GRID_MAIN))
cmp("narrow_proj_production_today", ref_b,
    ttnn.to_torch(T._narrow_proj_linear(zn_d, wb, ckc, ttnn.bfloat16)))
cmp("narrow_proj_l1_norm_l1_out", ref_b,
    ttnn.to_torch(T._narrow_proj_linear(zn_l, wb, ckc, ttnn.bfloat16, l1_out=True)))

print("=== trimul chain: proj -> proj -> multiply_ -> add_ ===", flush=True)
wg = ttnn.from_torch(torch.randn(C_Z, C_Z) * 0.05, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
z0 = torch.randn(1, TOK, TOK, C_Z) * 0.1


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
T._PAIR_PROJ_BW = 1
T._pair_proj_program_config.cache_clear()
chain_ref = chain(False)                       # the bit-exact DRAM reference contraction order
T._PAIR_PROJ_BW = 16
T._pair_proj_program_config.cache_clear()
cmp("trimul_chain_production_today", chain_ref, chain(False))
T._PAIR_PROJ_L1_OUT = True
cmp("trimul_chain_l1_out_bw1", chain_ref, chain(True))

Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print("wrote", sys.argv[1], flush=True)
