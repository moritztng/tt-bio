#!/usr/bin/env python3
"""Where an 8.31 ms Pairformer block spends its time at the 117-aa shape.

Step 3 of the multi-prediction throughput ladder. §1c of the state doc put the block at
~6.6 TFLOP/s while its clean matmuls run at 17-27% of peak, and named the difference the
layout tax. This ranks the block's parts by measured cost so the tax has an address.

Every timed region is bracketed by ttnn.synchronize_device: sync, t0, call, sync, t1. The
`pipe` column runs K calls between one pair of syncs, which is the in-model cost with
dispatch overlapped; `serial` is the same call fully drained. A part whose pipe is far
below its serial is dispatch-bound, and at this shape none of them are.

    TT_VISIBLE_DEVICES=3 python3 perf/trunk_layout/layer_split.py --n 128 --out split.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

from tt_bio.tenstorrent import get_device  # noqa: E402


def timeit(dev, fn, warm, iters, pipe):
    """(serial_ms, pipe_ms). fn() must be callable repeatedly with no shape drift."""
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ser = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ser.append((time.perf_counter() - t0) * 1e3)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(pipe):
        fn()
    ttnn.synchronize_device(dev)
    pip = (time.perf_counter() - t0) * 1e3 / pipe
    return sorted(ser)[len(ser) // 2], pip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=128, help="padded token count")
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--pipe", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    layer, c_z = build_layer(ckc)
    N = args.n
    torch.manual_seed(0)
    mk_s = lambda: ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    mk_z = lambda: ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    s, z = mk_s(), mk_z()
    print(f"layer built: c_z={c_z} N={N} hidden={layer.triangle_multiplication_start._hidden}",
          flush=True)

    rows = []

    def add(name, fn):
        ser, pip = timeit(dev, fn, args.warm, args.iters, args.pipe)
        rows.append(dict(part=name, serial_ms=round(ser, 4), pipe_ms=round(pip, 4)))
        print(f"  {name:34s} serial {ser:7.3f} ms   pipe {pip:7.3f} ms", flush=True)

    # whole block, the number everything else is a share of
    state = {"s": s, "z": z}

    def whole():
        state["s"], state["z"] = layer(state["s"], state["z"])
    add("BLOCK (s,z)", whole)

    z0 = mk_z()
    add("trimul start", lambda: layer.triangle_multiplication_start(z0, None))
    add("trimul end", lambda: layer.triangle_multiplication_end(z0, None))
    add("tri_att start", lambda: layer.triangle_attention_start(z0, None))
    add("tri_att end", lambda: layer.triangle_attention_end(z0, None))
    add("transition_z", lambda: layer.transition_z(z0))

    s0 = mk_s()
    sn = ttnn.layer_norm(s0, weight=layer.pre_norm_s_weight, bias=layer.pre_norm_s_bias,
                         epsilon=1e-5, compute_kernel_config=ckc)
    add("s attention_pair_bias", lambda: layer.attention_pair_bias(sn, z0, seq_mask=None))
    add("s transition", lambda: layer.transition_s(s0))

    # --- inside trimul: the per-channel-chunk layout ops -------------------------------
    tm = layer.triangle_multiplication_start
    from tt_bio.tenstorrent import (_triangle_mul_memory_config,
                                    _triangle_mul_program_config, _trimul_chunk_size)
    mc = _triangle_mul_memory_config(N)
    C_eff = _trimul_chunk_size(N, tm._hidden)
    pc = _triangle_mul_program_config((N + 31) // 32)
    rows.append(dict(part="trimul memory_config", note=str(mc.buffer_type), chunk=C_eff))
    print(f"  trimul buffer_type = {mc.buffer_type}, chunk = {C_eff}, "
          f"n_pairs = {tm._hidden // C_eff}", flush=True)

    zn = ttnn.layer_norm(z0, weight=tm.in_norm_weight, bias=tm.in_norm_bias, epsilon=1e-5,
                         compute_kernel_config=ckc)
    add("trimul: layer_norm(z)", lambda: ttnn.layer_norm(
        z0, weight=tm.in_norm_weight, bias=tm.in_norm_bias, epsilon=1e-5,
        compute_kernel_config=ckc))
    add("trimul: minimal_matmul x1", lambda: ttnn.experimental.minimal_matmul(
        zn, tm._gp_in_chunks(C_eff)[0], memory_config=mc, dtype=ttnn.bfloat16,
        compute_kernel_config=ckc))

    gp = ttnn.experimental.minimal_matmul(zn, tm._gp_in_chunks(C_eff)[0],
                                          memory_config=mc, dtype=ttnn.bfloat16,
                                          compute_kernel_config=ckc)
    add("trimul: chunk(4) x1", lambda: ttnn.chunk(gp, chunks=4, dim=-1))
    a0 = ttnn.chunk(gp, chunks=4, dim=-1)[2]
    add("trimul: permute(0,3,1,2) x1",
        lambda: ttnn.permute(a0, (0, 3, 1, 2), memory_config=mc))
    add("trimul: permute(0,3,2,1) x1",
        lambda: ttnn.permute(a0, (0, 3, 2, 1), memory_config=mc))
    ac = ttnn.permute(a0, (0, 3, 1, 2), memory_config=mc)
    bc = ttnn.permute(a0, (0, 3, 2, 1), memory_config=mc)
    add("trimul: chunk matmul x1", lambda: ttnn.matmul(
        ac, bc, compute_kernel_config=ckc, memory_config=mc, program_config=pc,
        dtype=ttnn.bfloat16))
    xc = ttnn.matmul(ac, bc, compute_kernel_config=ckc, memory_config=mc,
                     program_config=pc, dtype=ttnn.bfloat16)
    add("trimul: permute(0,2,3,1) x1",
        lambda: ttnn.permute(xc, (0, 2, 3, 1), memory_config=mc))
    xo = ttnn.permute(xc, (0, 2, 3, 1), memory_config=mc)
    add("trimul: reallocate x1", lambda: ttnn.reallocate(
        ttnn.permute(a0, (0, 3, 1, 2), memory_config=mc), memory_config=mc))

    # the running concat: cost of appending chunk i to an accumulator of i chunks
    C = xo.shape[-1]
    acc = ttnn.clone(xo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for i in (1, 5, 9):
        wide = ttnn.concat([acc] * i, dim=-1) if i > 1 else acc
        add(f"trimul: concat acc({i * C}ch) + {C}ch",
            lambda w=wide: ttnn.concat([w, xo], dim=-1))

    # --- inside tri_att: bias build vs attention ---------------------------------------
    ta = layer.triangle_attention_start
    z3 = ttnn.reshape(z0, tuple(z0.shape)[1:])
    add("tri_att: layer_norm(z)", lambda: ttnn.layer_norm(
        z3, weight=ta.layer_norm_weight, bias=ta.layer_norm_bias, epsilon=1e-5,
        compute_kernel_config=ckc))
    zl = ttnn.layer_norm(z3, weight=ta.layer_norm_weight, bias=ta.layer_norm_bias,
                         epsilon=1e-5, compute_kernel_config=ckc)
    add("tri_att: bias linear", lambda: ttnn.linear(
        zl, ta.bias_weight, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=8, x=8)))
    tb = ttnn.linear(zl, ta.bias_weight, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                     core_grid=ttnn.CoreGrid(y=8, x=8))
    tbu = ttnn.unsqueeze(tb, 0)
    add("tri_att: bias permute(0,3,1,2)", lambda: ttnn.permute(tbu, (0, 3, 1, 2)))
    add("tri_att: qkv linear", lambda: ttnn.linear(
        zl, ta.qkv_weight, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=8, x=8)))
    qkv = ttnn.linear(zl, ta.qkv_weight, compute_kernel_config=ckc,
                      core_grid=ttnn.CoreGrid(y=8, x=8))
    qkvu = ttnn.unsqueeze(qkv, 1)
    add("tri_att: nlp_create_qkv_heads", lambda: ttnn.experimental.nlp_create_qkv_heads(
        qkvu, num_heads=ta.n_heads, num_kv_heads=ta.n_heads, transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG))
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkvu, num_heads=ta.n_heads, num_kv_heads=ta.n_heads, transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bias = ttnn.permute(tbu, (0, 3, 1, 2))
    from tt_bio.tenstorrent import _tri_att_sdpa_program_config
    add("tri_att: SDPA", lambda: ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=ta.scale ** -1,
        program_config=_tri_att_sdpa_program_config(q.shape[2], k.shape[2])))
    o = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=ta.scale ** -1,
        program_config=_tri_att_sdpa_program_config(q.shape[2], k.shape[2]))
    add("tri_att: nlp_concat_heads", lambda: ttnn.experimental.nlp_concat_heads(
        o, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    oc = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(
        o, memory_config=ttnn.DRAM_MEMORY_CONFIG), 1)
    add("tri_att: out linear", lambda: ttnn.linear(
        oc, ta.o_weight, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=8, x=8)))

    block = next(r["pipe_ms"] for r in rows if r["part"] == "BLOCK (s,z)")
    print(f"\nblock {block:.3f} ms; part shares:", flush=True)
    for r in rows:
        if "pipe_ms" in r and r["part"] != "BLOCK (s,z)":
            r["share_of_block"] = round(r["pipe_ms"] / block, 4)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            dict(n=N, c_z=c_z, block_pipe_ms=block, rows=rows), indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
