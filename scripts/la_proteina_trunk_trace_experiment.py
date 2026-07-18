#!/usr/bin/env python3
# La-Proteina trunk trace-capture experiment (pass 6 perf pass).
#
# Tests whether the 14-layer denoiser trunk is host-dispatch-bound (trace helps)
# or compute-bound (trace ~0) at N=64. Captures the trunk as a trace and
# compares eager vs trace-replay wall-clock. Per the tt-porting trace knowledge,
# large-protein trunks are compute-bound (trace ~0); small proteins can be
# dispatch-bound (~30% win). This experiment decides which case La-Proteina
# N=64 is in.
#
#   TT_VISIBLE_DEVICES=0 python3 scripts/la_proteina_trunk_trace_experiment.py
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(WORKTREE / "scripts"))
sys.path.insert(0, str(WORKTREE))

import la_proteina_sampler_loop_parity as parity  # noqa: E402

B, N = parity.B, parity.N
DTYPE = os.environ.get("LA_PROTEINA_DTYPE", "bf16")


def main():
    mask = torch.ones(B, N, dtype=torch.bool)
    mask_f = mask.float()
    pair_mask = mask[:, :, None] * mask[:, None, :]

    dev = ttnn.open_device(device_id=0, trace_region_size=200_000_000)
    try:
        arch = dev.arch()
        ck = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        dev.enable_program_cache()
        dtype = ttnn.bfloat16 if DTYPE == "bf16" else ttnn.float32

        def to_tt(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)

        mask_tt = to_tt(mask_f.unsqueeze(-1))
        pair_mask_tt = to_tt(pair_mask.float().unsqueeze(-1))
        pmb = torch.zeros(B, 1, N, N)
        pmb = torch.where(pair_mask.unsqueeze(1), pmb, torch.full_like(pmb, -1e4))
        pmb_tt = to_tt(pmb)

        torch.manual_seed(1234)
        g_nn = parity.LocalLatentsTransformer(**parity.CFG).eval()
        port_sd = parity.build_port_sd(g_nn)
        denoiser = parity.TTLaProteinaDenoiser(
            dev, ck, port_sd, parity.CFG, parity.FEAT_DIMS, dtype=dtype,
            factory_dtype=dtype,
        )

        # build representative inputs (random, fixed)
        def rand_tt(*shape):
            return to_tt(torch.randn(*shape, dtype=torch.float32))

        seqs = rand_tt(B, N, parity.CFG["token_dim"])
        pair_rep = rand_tt(B, N, N, parity.CFG["pair_repr_dim"])
        c_pre = rand_tt(B, N, parity.CFG["dim_cond"])

        def trunk():
            return denoiser.trunk(seqs, pair_rep, c_pre, mask_tt, pmb_tt)

        # warmup (prime program cache)
        trunk(); ttnn.synchronize_device(dev)
        trunk(); ttnn.synchronize_device(dev)

        # eager timing
        def time_fn(fn, repeats=10):
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn()
                ttnn.synchronize_device(dev)
                ts.append(time.perf_counter() - t0)
            ts.sort()
            return ts[len(ts) // 2] * 1e3

        eager_ms = time_fn(trunk)
        print(f"trunk eager: {eager_ms:.2f} ms")

        # trace capture
        ttnn.synchronize_device(dev)
        trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        out = trunk()
        ttnn.end_trace_capture(dev, trace_id, cq_id=0)
        ttnn.synchronize_device(dev)

        def replay():
            ttnn.execute_trace(dev, trace_id, cq_id=0, blocking=True)
            return out

        # warmup replay
        replay(); ttnn.synchronize_device(dev)
        trace_ms = time_fn(replay)
        print(f"trunk trace replay: {trace_ms:.2f} ms")
        print(f"trace speedup: {eager_ms/trace_ms:.2f}x  (delta {eager_ms - trace_ms:.2f} ms)")

        ttnn.release_trace(dev, trace_id)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
