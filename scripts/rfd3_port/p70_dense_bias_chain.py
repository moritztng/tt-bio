#!/usr/bin/env python3
"""p70 -- the L2 screen: what is the DiT's dense scores+bias chain actually worth?

The chain, `RFD3AtomBlock.__call__` on the dense branch (`sparse_qk is None`, which is what
`LocalTokenTransformer` takes), six ops between the hoisted pair-bias projection and the softmax:

    pair_bias = permute(pair_bias, (0, 3, 1, 2))            # [1,I,J,H] -> [1,H,I,J]   bf16
    bias      = pad(add(pair_bias, additive_mask), J->n_key, -1e4)                     bf16
    bias_f    = typecast(bias, fp32)
    scores_f  = typecast(scores, fp32)
    scores    = add(scores_f, bias_f, a_activations=[MUL_UNARY_SFPU(scale)])            fp32

`tt_bio/rfd3_bias.py:fused_scores_bias_fp32` already does exactly this for the SPARSE atom path
(reads bf16 scores + a compact [1,H,L,K] bias, writes scores*scale+bias in fp32, dense bias never
in DRAM). The dense branch never got it. This screen prices the swap BEFORE the reader is written.

Two things it must not do, both from the memories:

  * not oversync. `tt-bio-isolated-op-timing-oversync-inflates-cost`: p49's per-op numbers are ~2x
    over-reads. Arm A issues the six ops back-to-back the way the block does and syncs once.
  * not assert a roof. `roofline-roof-must-be-measured-not-asserted`: the fused kernel's cost is
    predicted from arm B, a MEASURED op that moves almost exactly the kernel's bytes
    (`typecast(scores_bf16 -> fp32)`: r 15.4 + w 30.9 MB against the kernel's r 31.8 + w 30.9),
    plus the compact bias read at arm C's measured read rate. No 102 TFLOP/s, no 390 GB/s asserted.

Gate, written before the run: **net >= 10 ms/step or NO-GO** (state/rfd3-b8-to-4x-p2.md §3).

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p2 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p2 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p70_dense_bias_chain.py \
          perf/p70/dense_bias_chain.json
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p70/dense_bias_chain.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6

I = 685            # tokens at the page fixture (9q6y chain A, 585 target + 100 binder)
J = 685
H = 16             # n_head of the DiT
NKEY = 704         # _align_tile(685)
HEAD_DIM = 48
SCALE = HEAD_DIM ** -0.5
CALLS_PER_STEP = 36        # 18 blocks x 2 recycles
MB = 1024.0 * 1024.0


def mb(*tensors):
    tot = 0.0
    for t in tensors:
        n = 1
        for d in t.padded_shape:
            n *= int(d)
        tot += n * (4 if t.dtype == ttnn.float32 else 2)
    return tot / MB


def timeit(fn, n=N, warm=2):
    """One sync per rep, nothing inside. Returns (median ms, all ms)."""
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(DEV)
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out), [round(v, 4) for v in out]


def main():
    global DEV
    # `get_device` is the only correct opener here: a bare ttnn.open_device with
    # TT_VISIBLE_DEVICES set to one card hits "Custom fabric mesh graph descriptor path must be
    # specified for CUSTOM cluster type", and it is also what takes the host-local card lease.
    DEV = get_device()
    run()


def run():
    torch.manual_seed(0)
    tt = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(          # noqa: E731
        x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # exactly the tensors the dense branch holds at this fixture
    pair_bias0 = tt(torch.randn(1, I, J, H) * 0.5)             # [1,I,J,H] projection output
    mask = tt(torch.where(torch.rand(1, 1, I, J) < 0.05, -1e4, 0.0))
    scores = tt(torch.randn(1, H, I, NKEY) * 0.3)              # bf16 QK output, key axis padded

    res = {"shape": {"I": I, "J": J, "H": H, "n_key": NKEY, "head_dim": HEAD_DIM},
           "calls_per_step": CALLS_PER_STEP, "n": N, "arms": {}}

    def chain():
        pb = ttnn.permute(pair_bias0, (0, 3, 1, 2))
        s = ttnn.add(pb, mask)
        pad = [(0, 0)] * 4
        pad[3] = (0, NKEY - J)
        b = ttnn.pad(s, pad, -1e4)
        bf = ttnn.typecast(b, ttnn.float32, memory_config=b.memory_config())
        sf = ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config())
        o = ttnn.add(sf, bf, input_tensor_a_activations=[
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SCALE)])
        for t in (pb, s, b, bf, sf, o):
            ttnn.deallocate(t)

    a_ms, a_all = timeit(chain)
    res["arms"]["A_chain_6op"] = {"ms_per_call": round(a_ms, 4), "all": a_all,
                                  "ms_per_step": round(a_ms * CALLS_PER_STEP, 3)}
    print("[p70] A  six-op chain      %8.4f ms/call -> %7.3f ms/step" %
          (a_ms, a_ms * CALLS_PER_STEP), flush=True)

    # Arm B -- the measured proxy for the fused kernel's cost. typecast(bf16 scores -> fp32)
    # moves r 15.4 + w 30.9 MB; the kernel moves r 31.8 + w 30.9. Same write, one extra bf16
    # read of the same size as the first, streamed row-block at a time into L1.
    def proxy():
        o = ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config())
        ttnn.deallocate(o)

    b_ms, b_all = timeit(proxy)
    b_bytes = mb(scores) + 2 * mb(scores)
    res["arms"]["B_typecast_proxy"] = {"ms_per_call": round(b_ms, 4), "all": b_all,
                                       "mb_moved": round(b_bytes, 2),
                                       "gb_s": round(b_bytes / 1024.0 / (b_ms / 1e3), 1)}
    print("[p70] B  typecast proxy    %8.4f ms/call  (%.1f MB, %.1f GB/s)" %
          (b_ms, b_bytes, b_bytes / 1024.0 / (b_ms / 1e3)), flush=True)

    # Arm C -- a pure bf16 read+write of the compact bias's size, to price the extra read.
    def readc():
        o = ttnn.clone(pair_bias0)
        ttnn.deallocate(o)

    c_ms, c_all = timeit(readc)
    c_bytes = 2 * mb(pair_bias0)
    res["arms"]["C_bias_clone"] = {"ms_per_call": round(c_ms, 4), "all": c_all,
                                   "mb_moved": round(c_bytes, 2),
                                   "gb_s": round(c_bytes / 1024.0 / (c_ms / 1e3), 1)}
    print("[p70] C  compact clone     %8.4f ms/call  (%.1f MB, %.1f GB/s)" %
          (c_ms, c_bytes, c_bytes / 1024.0 / (c_ms / 1e3)), flush=True)

    # Arm D -- the shipped fp32 scaled add alone, for attribution inside arm A.
    sf = ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config())
    bf = ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config())

    def add_only():
        o = ttnn.add(sf, bf, input_tensor_a_activations=[
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SCALE)])
        ttnn.deallocate(o)

    d_ms, d_all = timeit(add_only)
    res["arms"]["D_fp32_scaled_add"] = {"ms_per_call": round(d_ms, 4), "all": d_all,
                                        "mb_moved": round(3 * mb(sf), 2)}
    print("[p70] D  fp32 scaled add   %8.4f ms/call  (%.1f MB)" % (d_ms, 3 * mb(sf)), flush=True)
    ttnn.deallocate(sf)
    ttnn.deallocate(bf)

    # the prediction. read side of the extra compact bias at arm C's measured read rate
    # (half of C's bytes are the read).
    extra_read_ms = c_ms / 2.0
    kernel_ms = b_ms + extra_read_ms
    net_step = (a_ms - kernel_ms) * CALLS_PER_STEP
    res["prediction"] = {
        "kernel_ms_per_call": round(kernel_ms, 4),
        "kernel_basis": "arm B (same write, first read) + half arm C (the compact bias read)",
        "net_ms_per_step": round(net_step, 3),
        "net_s_per_design_200_steps": round(net_step * 200 / 1e3, 3),
        "gate_ms_per_step": 10.0,
        "verdict": "GO" if net_step >= 10.0 else "NO-GO",
    }
    print("\n[p70] fused kernel predicted %.4f ms/call (B %.4f + C/2 %.4f)"
          % (kernel_ms, b_ms, extra_read_ms))
    print("[p70] net %+.3f ms/step = %+.3f s/design at 200 steps   gate >= 10 ms/step -> %s"
          % (net_step, net_step * 200 / 1e3, res["prediction"]["verdict"]), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
