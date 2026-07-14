"""Re-check PairWeightedAveraging with all heads packed into four device ops.

Uses real Protenix-v2 weights and production MSA dimensions.  The candidate is
mathematically equivalent to the per-head production path:

* project pair logits, MSA values, and gates for every head at once;
* execute one head-batched weighted-average matmul;
* concatenate heads and execute one output projection instead of projecting
  each head separately and summing.

The script is read-only with respect to the runtime path.  It reports warm
device-synchronized timings and output parity.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import ttnn

from boltz2_protenix_kernel_scout import _build_pwa, _compare, _device_setup, _load_checkpoint


def packed(module, m_in, z_in):
    m = ttnn.reshape(m_in, tuple(m_in.shape)[1:])
    z = ttnn.reshape(z_in, tuple(z_in.shape)[1:])
    m = ttnn.layer_norm(
        m,
        weight=module.m_norm_weight,
        bias=module.m_norm_bias,
        epsilon=1e-5,
        compute_kernel_config=module.compute_kernel_config,
    )
    z = ttnn.layer_norm(
        z,
        weight=module.z_norm_weight,
        bias=module.z_norm_bias,
        epsilon=1e-5,
        compute_kernel_config=module.compute_kernel_config,
    )

    # [N,N,H] -> [H,N,N], one softmax matrix per head.
    w = ttnn.linear(
        z,
        module.z_weight,
        compute_kernel_config=module.compute_kernel_config,
        core_grid=ttnn.CoreGrid(y=module.device.compute_with_storage_grid_size().y,
                                x=module.device.compute_with_storage_grid_size().x),
    )
    ttnn.deallocate(z)
    w = ttnn.permute(w, (2, 0, 1))
    w = ttnn.softmax(
        w, dim=-1, compute_kernel_config=module.compute_kernel_config, numeric_stable=True
    )

    # [M,N,H*D] -> [H,M*D,N].  Folding M into the matmul row dimension
    # avoids replicating the [H,N,N] weights across the MSA batch.
    msa_depth, seq_len = m.shape[0], m.shape[1]
    v = ttnn.linear(
        m,
        module.m_weight,
        compute_kernel_config=module.compute_kernel_config,
        core_grid=ttnn.CoreGrid(y=module.device.compute_with_storage_grid_size().y,
                                x=module.device.compute_with_storage_grid_size().x),
    )
    v = ttnn.reshape(v, (msa_depth, seq_len, module.n_heads, module.head_dim))
    v = ttnn.permute(v, (2, 0, 3, 1))
    v = ttnn.reshape(
        v, (module.n_heads, msa_depth * module.head_dim, seq_len)
    )
    o = ttnn.matmul(
        v,
        w,
        transpose_b=True,
        compute_kernel_config=module.compute_kernel_config,
        core_grid=ttnn.CoreGrid(y=module.device.compute_with_storage_grid_size().y,
                                x=module.device.compute_with_storage_grid_size().x),
    )
    ttnn.deallocate(v)
    ttnn.deallocate(w)
    o = ttnn.reshape(
        o, (module.n_heads, msa_depth, module.head_dim, seq_len)
    )
    o = ttnn.permute(o, (1, 0, 3, 2))

    g = ttnn.linear(
        m,
        module.g_weight,
        compute_kernel_config=module.compute_kernel_config,
        core_grid=ttnn.CoreGrid(y=module.device.compute_with_storage_grid_size().y,
                                x=module.device.compute_with_storage_grid_size().x),
    )
    ttnn.deallocate(m)
    g = ttnn.reshape(g, (msa_depth, seq_len, module.n_heads, module.head_dim))
    g = ttnn.permute(g, (0, 2, 1, 3))
    o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(g)

    # Sum(head_output @ W_head) == concat(head_output) @ concat(W_head).
    o = ttnn.permute(o, (0, 2, 1, 3))
    o = ttnn.reshape(o, (msa_depth, seq_len, module.n_heads * module.head_dim))
    out = ttnn.linear(
        o,
        module.o_weight,
        compute_kernel_config=module.compute_kernel_config,
        core_grid=ttnn.CoreGrid(y=module.device.compute_with_storage_grid_size().y,
                                x=module.device.compute_with_storage_grid_size().x),
    )
    ttnn.deallocate(o)
    return ttnn.reshape(out, (1, *out.shape))


def packed_inputs(module, m_in, z_in):
    """Pack only the three repeatedly-read input projections.

    Keep the production per-head matmul and output reduction so this isolates
    whether eliminating seven redundant reads of each large input can win
    without the 4-D batched-matmul layout cost of ``packed``.
    """
    m = ttnn.reshape(m_in, tuple(m_in.shape)[1:])
    z = ttnn.reshape(z_in, tuple(z_in.shape)[1:])
    m = ttnn.layer_norm(
        m,
        weight=module.m_norm_weight,
        bias=module.m_norm_bias,
        epsilon=1e-5,
        compute_kernel_config=module.compute_kernel_config,
    )
    z = ttnn.layer_norm(
        z,
        weight=module.z_norm_weight,
        bias=module.z_norm_bias,
        epsilon=1e-5,
        compute_kernel_config=module.compute_kernel_config,
    )
    grid = ttnn.CoreGrid(
        y=module.device.compute_with_storage_grid_size().y,
        x=module.device.compute_with_storage_grid_size().x,
    )
    b_all = ttnn.linear(
        z, module.z_weight, compute_kernel_config=module.compute_kernel_config, core_grid=grid
    )
    ttnn.deallocate(z)
    v_all = ttnn.linear(
        m, module.m_weight, compute_kernel_config=module.compute_kernel_config, core_grid=grid
    )
    g_all = ttnn.linear(
        m, module.g_weight, compute_kernel_config=module.compute_kernel_config, core_grid=grid
    )
    ttnn.deallocate(m)
    bs = ttnn.chunk(b_all, module.n_heads, dim=-1)
    vs = ttnn.chunk(v_all, module.n_heads, dim=-1)
    gs = ttnn.chunk(g_all, module.n_heads, dim=-1)
    ttnn.deallocate(b_all)
    ttnn.deallocate(v_all)
    ttnn.deallocate(g_all)

    out = None
    for i, (b, v, g) in enumerate(zip(bs, vs, gs)):
        b = ttnn.permute(b, (2, 0, 1))
        w = ttnn.softmax(
            b, dim=-1, compute_kernel_config=module.compute_kernel_config, numeric_stable=True
        )
        v = ttnn.permute(v, (0, 2, 1))
        o = ttnn.matmul(
            v,
            w,
            transpose_b=True,
            compute_kernel_config=module.compute_kernel_config,
            core_grid=grid,
        )
        ttnn.deallocate(v)
        ttnn.deallocate(w)
        o = ttnn.permute(o, (0, 2, 1))
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        o = ttnn.linear(
            o,
            module.o_weight[i * module.head_dim : (i + 1) * module.head_dim, :],
            compute_kernel_config=module.compute_kernel_config,
            core_grid=grid,
        )
        if out is None:
            out = o
        else:
            previous = out
            out = ttnn.add(previous, o)
            ttnn.deallocate(previous)
            ttnn.deallocate(o)
    return ttnn.reshape(out, (1, *out.shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--msa-depth", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checkpoint", default="/home/moritz/.boltz/protenix-v2.pt")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20260714)

    state = _load_checkpoint(args.checkpoint)
    _, device, config = _device_setup()
    modules = _build_pwa(state, config)
    module = modules[0]
    c_m = int(module.weights["proj_m.weight"].shape[1])
    c_z = int(module.weights["proj_z.weight"].shape[1])

    def run(fn, m, z, capture=False):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = fn(module, m, z) if fn is not None else module(m, z)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        host = torch.Tensor(ttnn.to_torch(out)).float() if capture else None
        ttnn.deallocate(out)
        return elapsed, host

    for n in args.sizes:
        gen = torch.Generator().manual_seed(20260714 + n)
        m_host = torch.randn(
            (1, args.msa_depth, n, c_m), generator=gen, dtype=torch.bfloat16
        )
        z_host = torch.randn((1, n, n, c_z), generator=gen, dtype=torch.bfloat16)
        m = ttnn.from_torch(
            m_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
        )
        z = ttnn.from_torch(
            z_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
        )
        run(None, m, z)
        run(packed, m, z)
        run(packed_inputs, m, z)
        baseline = [run(None, m, z)[0] for _ in range(args.repeats)]
        candidate = [run(packed, m, z)[0] for _ in range(args.repeats)]
        input_candidate = [
            run(packed_inputs, m, z)[0] for _ in range(args.repeats)
        ]
        _, ref = run(None, m, z, capture=True)
        _, got = run(packed, m, z, capture=True)
        _, got_inputs = run(packed_inputs, m, z, capture=True)
        base_s = statistics.median(baseline)
        packed_s = statistics.median(candidate)
        packed_inputs_s = statistics.median(input_candidate)
        print(
            {
                "N": n,
                "msa_depth": args.msa_depth,
                "baseline_ms": base_s * 1000,
                "packed_ms": packed_s * 1000,
                "speedup": base_s / packed_s,
                "parity": _compare(ref, got),
                "packed_inputs_ms": packed_inputs_s * 1000,
                "packed_inputs_speedup": base_s / packed_inputs_s,
                "packed_inputs_parity": _compare(ref, got_inputs),
                "baseline_samples_ms": [x * 1000 for x in baseline],
                "packed_samples_ms": [x * 1000 for x in candidate],
                "packed_inputs_samples_ms": [x * 1000 for x in input_candidate],
            },
            flush=True,
        )
        ttnn.deallocate(m)
        ttnn.deallocate(z)


if __name__ == "__main__":
    main()
