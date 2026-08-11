#!/usr/bin/env python3
"""triatt-fused-kernel-final, planning pass: the reference numbers every Phase-2 gate compares
against, re-measured on this card, plus the one block knob nobody has swept.

Card 2 on qb2, P300c, 11x10 = 110 cores, ttnn 0.68.0, branch wk/triatt-fused-kernel-final.
Model shape: protenix-v2 Pairformer, 512 aa, c_z = 256, 8 heads, head_dim 32.

PREDICTIONS, WRITTEN BEFORE THE RUN (state/triatt-fused-kernel-final.md 4):

R1  minimal_matmul qkv (M = 262144, K = 256, N = 768, DRAM->DRAM, `_qkv_mm_config`) lands
    2.05-2.20 ms, i.e. reproduces the 2.103 ms the prior pass measured to within 5 %.
    This is the number the generic_op re-drive of the same op must match in Phase 2 step 0.
R2  nlp_create_qkv_heads on that output lands 2.10-2.25 ms (prior 2.154). This is the whole
    prize of the head-major writer: it is deleted, not accelerated.
R3  SDPA q_chunk 512 / k_chunk 256, bias on, lands 6.35-6.65 ms (prior 6.496) and bias off
    2.20-2.40 (prior 2.285). Their ratio is the bias re-read and it is the K2 prize.
R4  nlp_concat_heads 0.70-0.80 (prior 0.740); gate multiply_ 1.00-1.10 (prior 1.039);
    out through _pair_proj_linear 0.75-0.85 (prior 0.784, the landed minimal_matmul leg).
R5  The shipped `_MM_BLOCK[24]` entry (4,8,1,4,1) is NOT the best blocking for the qkv shape.
    moonshot-4x-k256-kernel-rate 3c measured (4,8,2,4,1) best on a K=256 N=768 DRAM-out arm
    at 1.116x over that op's internal default, and nobody has ever compared the two directly
    at this shape. I predict the shipped entry is within 1.00-1.08x of the best legal config
    here, i.e. there is a small free win but not a large one, and that every config with
    K_block_size = 8 is torch.equal to every other (K stays one accumulation block).
R6  Every arm above is stable to <= 3 % A/A on a quiet box. Anything wider means a co-tenant
    is running and no ranking inside 1.1x may be pinned off this file.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T

MiB = 2 ** 20
RES = {"predictions": __doc__, "arms": []}


def timed(fn, warm=2, reps=5):
    dev = T.get_device()
    for _ in range(warm):
        r = fn()
        del r
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        del r
    return st.median(ts), (max(ts) - min(ts)) / st.median(ts)


def add(label, fn, **extra):
    try:
        ms, aa = timed(fn)
        row = {"arm": label, "ms": ms * 1e3, "aa_spread": aa}
        row.update(extra)
    except Exception as e:  # noqa: BLE001 - a failed arm must not lose the file
        row = {"arm": label, "error": repr(e)[:200]}
    row.update(extra)
    RES["arms"].append(row)
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default="perf/triatt_fused/step0_baseline.json")
    args = ap.parse_args()
    S, C, H, D = args.n, 256, 8, 32

    dev = T.get_device()
    ckc = T.TorchWrapper.compute_kernel_config if hasattr(T.TorchWrapper, "compute_kernel_config") else None
    if ckc is None or not isinstance(ckc, ttnn.DeviceComputeKernelConfig):
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
    RES["meta"] = {"n": S, "c": C, "heads": H, "head_dim": D,
                   "grid": list(T.COMPUTE_GRID_MAIN), "loadavg": os.getloadavg()}
    print(json.dumps(RES["meta"]), flush=True)

    def dram(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x = dram(torch.randn(S, S, C, dtype=torch.float32).to(torch.bfloat16))
    w_qkv = dram(torch.randn(C, 3 * H * D, dtype=torch.float32).to(torch.bfloat16))
    w_g = dram(torch.randn(C, H * D, dtype=torch.float32).to(torch.bfloat16))
    w_o = dram(torch.randn(C, C, dtype=torch.float32).to(torch.bfloat16))

    cfg = T._qkv_mm_config(x, w_qkv)
    RES["meta"]["qkv_mm_config"] = str(cfg)

    def mm(inp, w, c):
        return ttnn.experimental.minimal_matmul(
            input_tensor=inp, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=c)

    add("R1_qkv_minimal_matmul", lambda: mm(x, w_qkv, cfg), shape=f"{S*S}x{C}@{C}x{3*H*D}")
    add("R4_gate_minimal_matmul", lambda: mm(x, w_g, T._qkv_mm_config(x, w_g)),
        shape=f"{S*S}x{C}@{C}x{H*D}")

    qkv = mm(x, w_qkv, cfg)
    qkv_u = ttnn.unsqueeze(qkv, 1)

    def heads():
        return ttnn.experimental.nlp_create_qkv_heads(
            qkv_u, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
            memory_config=qkv_u.memory_config())

    add("R2_nlp_create_qkv_heads", heads)
    q, k, v = heads()
    ttnn.deallocate(qkv)

    bias = dram(torch.randn(1, H, S, S, dtype=torch.float32).to(torch.bfloat16))
    scale = D ** -0.5
    for qc, kc, lab in ((512, 256, "wideq"), (256, 256, "shipped")):
        pc = T._sdpa_program_config(qc, kc)
        add(f"R3_sdpa_bias_on_{lab}", lambda pc=pc: ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=scale, program_config=pc),
            q_chunk=qc, k_chunk=kc)
        add(f"R3_sdpa_bias_off_{lab}", lambda pc=pc: ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, scale=scale, program_config=pc),
            q_chunk=qc, k_chunk=kc)

    o = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=scale,
        program_config=T._sdpa_program_config(512, 256))
    for t in (q, k, v, bias):
        ttnn.deallocate(t)
    add("R4_nlp_concat_heads",
        lambda: ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    oc = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG), 1)
    ttnn.deallocate(o)

    g = mm(x, w_g, T._qkv_mm_config(x, w_g))
    add("R4_gate_multiply", lambda: ttnn.multiply(
        oc, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
        memory_config=ttnn.DRAM_MEMORY_CONFIG))
    add("R4_out_pair_proj",
        lambda: T._pair_proj_linear(oc, w_o, ckc, ttnn.bfloat16, l1_out=True))
    for t in (g, oc):
        ttnn.deallocate(t)

    # R5: the never-swept nt=24 block entry, on the shape the fold issues, parity per config.
    ref = ttnn.to_torch(mm(x, w_qkv, cfg))
    mt = (S * S) // 32
    for M in (2, 4, 8):
        for N in (1, 2, 3):
            for sh, sw in ((4, 1), (2, 1), (1, 1)):
                if mt % M or 24 % N or sh * sw > 4:
                    continue
                c = ttnn.MinimalMatmulConfig(
                    M_block_size=M, K_block_size=8, N_block_size=N, subblock_h=sh,
                    subblock_w=sw, compute_with_storage_grid_size=T._mm_core_coord(*T.COMPUTE_GRID_MAIN))
                row = add(f"R5_mm24_M{M}_N{N}_s{sh}x{sw}", lambda c=c: mm(x, w_qkv, c),
                          M_block=M, N_block=N, sub=f"{sh}x{sw}")
                if "error" not in row:
                    try:
                        row["equal_to_shipped"] = bool(torch.equal(ttnn.to_torch(mm(x, w_qkv, c)), ref))
                    except Exception as e:  # noqa: BLE001
                        row["equal_to_shipped"] = repr(e)[:120]
                    print(json.dumps(row), flush=True)

    RES["meta"]["loadavg_end"] = os.getloadavg()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RES, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
