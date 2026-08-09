#!/usr/bin/env python3
"""The eight T1 ops at the shapes a LIVE 298 aa fold issues them at.

`pf_block_ops.py` (and W1's whole ledger) builds the block with z = [1, N, N, c_z] at N=320, but a
real fold carries the pair tensor as [298, 320, 256]: TILE_LAYOUT pads only the last two dims, so
the row axis stays at 298 tokens while the column axis pads 298 -> 320. The harness block is
therefore 320/298 = 1.074x heavier on the row axis than the fold's. Call counts per fold are from
perf/tri_att_298/blockcount_298_pv2.json, measured in a live fold.
"""
import json, statistics as st, sys, time
from pathlib import Path
import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device, _pair_proj_linear, _tri_att_sdpa_program_config  # noqa: E402

DEV = get_device()
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
M, N, CZ, NH, HD = 298, 320, 256, 8, 32
ckc = ttnn.init_device_compute_kernel_config(
    DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn, warm=2, pipe=3, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(DEV)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def T(shape, mc=DRAM):
    return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=DEV,
                           dtype=ttnn.bfloat16, memory_config=mc)


R = {"shape_note": "pair tensor [298, 320, 256]; q/k/v [298, 8, 320, 32]; bias [1, 8, 320, 320]"}
x = T((M, N, CZ))
q, k, v = (T((M, NH, N, HD)) for _ in range(3))
mb = T((1, NH, N, N))
prod = _tri_att_sdpa_program_config(N, N)


def sdpa(mask, cfg=prod):
    return ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=False, scale=HD ** -0.5, program_config=cfg)


R["sdpa_prod_us"] = round(timed(lambda: ttnn.deallocate(sdpa(mb))) * 1e6, 1)
R["sdpa_nomask_us"] = round(timed(lambda: ttnn.deallocate(sdpa(None))) * 1e6, 1)
R["sdpa_chunk320_us"] = round(timed(lambda: ttnn.deallocate(sdpa(
    ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False,
                           q_chunk_size=320, k_chunk_size=320) and mb,
    ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False,
                           q_chunk_size=320, k_chunk_size=320)))) * 1e6, 1)
try:
    mm = T((M, NH, N, N))
    R["sdpa_materialised_mask_us"] = round(timed(lambda: ttnn.deallocate(sdpa(mm)), pipe=2) * 1e6, 1)
    ttnn.deallocate(mm)
except Exception as e:                                                   # noqa: BLE001
    R["sdpa_materialised_err"] = str(e)[:120]

qkv_in = T((M, 1, N, 3 * NH * HD))
R["heads_us"] = round(timed(lambda: [ttnn.deallocate(o) for o in ttnn.experimental.nlp_create_qkv_heads(
    qkv_in, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=DRAM)]) * 1e6, 1)
ttnn.deallocate(qkv_in)

R["permute_dram_us"] = round(timed(lambda: ttnn.deallocate(ttnn.permute(x, (1, 0, 2)))) * 1e6, 1)
R["permute_l1_us"] = round(timed(lambda: ttnn.deallocate(
    ttnn.permute(x, (1, 0, 2), memory_config=L1))) * 1e6, 1)
R["clone_dram_us"] = round(timed(lambda: ttnn.deallocate(ttnn.clone(x, memory_config=DRAM))) * 1e6, 1)

for lbl, nout in (("qkv", 3 * NH * HD), ("gate", CZ)):
    w = T((CZ, nout))
    R[lbl + "_us"] = round(timed(lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
        input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=ttnn.bfloat16))) * 1e6, 1)
    ttnn.deallocate(w)

wb = T((CZ, 32))
R["bias_us"] = round(timed(lambda: ttnn.deallocate(
    _pair_proj_linear(x, wb, ckc, ttnn.bfloat16))) * 1e6, 1)
ttnn.deallocate(wb)

a, b = T((M, N, CZ)), T((M, N, CZ))
R["gate_multiply_us"] = round(timed(lambda: ttnn.multiply_(
    a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])) * 1e6, 1)

print(json.dumps(R, indent=1))
json.dump(R, open(REPO / "perf/tri_att_298/trueshape_c2.json", "w"), indent=1)
