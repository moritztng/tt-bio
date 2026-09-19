#!/usr/bin/env python3
"""Throughput of the taped SHIPPED forward against the taped primitive twin.

The row's question is whether making the fast path differentiable actually buys the fast
path's speed, and a correct differentiable fast path that is not faster is a negative
result. So both arms are measured, interleaved, on the same shapes and the same weights.

  SHIPPED  the production `tt_bio.tenstorrent.Transition` under `ag.tape()` -- its row
           chunking, its L1 routings, its fused silu, its program configs, CORE_GRID_MAIN.
  TWIN     the same maths composed from `tt_bio.autograd` primitives, which is what the
           deleted `perf/ptxft/tape_block.py` did and what `perf/hallgrad/e2e_distogram.py`
           still does. It is the surviving form of the thing the charter calls "the slow
           differentiable twin"; the charter's own twin is gone (`state/ptx/LEDGER.md` R2)
           and `_RECIPES` now holds only LoRA on a frozen trunk, which computes no trunk
           gradient at all, so this composition is the only second arm there is.

Both arms run the SAME backward -- `_taped_linear`, `_taped_layer_norm` and the tape's
eltwise rules -- so the difference measured is the forward, which is the whole claim.

Arms interleave one-for-one rather than running in blocks, because a block schedule
attributes any drift in clock or co-tenancy to whichever arm went second. AICLK is sampled
DURING and reported as a range; a number without one is not a measurement on this part.

`--mode inference` prints a digest of the untaped shipped forward and nothing else. Run it
under two trees and diff: that is the byte-identical control `state/ptx/LEDGER.md` K13 asks
for before anything is routed on the assumption it holds.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch


def clocks(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                               capture_output=True, text=True, timeout=25)
            for d in json.loads(r.stdout).get("device_info", []):
                c = d.get("telemetry", {}).get("aiclk")
                if c is not None:
                    out.append(int(c))
        except Exception:
            pass
        time.sleep(0.25)


def weights(rng, c, hidden):
    return {
        "norm.weight": torch.tensor(rng.standard_normal(c) * 0.3 + 1.0, dtype=torch.float32),
        "norm.bias": torch.tensor(rng.standard_normal(c) * 0.1, dtype=torch.float32),
        "fc1.weight": torch.tensor(rng.standard_normal((hidden, c)) / math.sqrt(c),
                                   dtype=torch.float32),
        "fc2.weight": torch.tensor(rng.standard_normal((hidden, c)) / math.sqrt(c),
                                   dtype=torch.float32),
        "fc3.weight": torch.tensor(rng.standard_normal((c, hidden)) / math.sqrt(hidden),
                                   dtype=torch.float32),
    }


def attention_ab(a, ttnn, tt, ag):
    """The fused SDPA under the tape, against the tape's own composed attention.

    This is where the row's claim lives. `Transition`'s tuning is mostly L1 placement, and
    the tape has to give L1 back; triangle attention's tuning is a genuinely different
    kernel -- one fused pass against matmul, softmax, matmul with the score tensor
    materialised -- and the tape keeps that kernel for the forward. Both arms run the SAME
    backward, `autograd.triangle_attention`'s chunked recompute, so the only difference
    measured is the forward.
    """
    from tt_bio import taped_ttnn as tp
    B, H, N, dh = a.tokens, a.heads, a.tokens, a.head_dim
    B = a.tokens
    scale = dh ** -0.5
    rng = np.random.default_rng(0)
    dev = tt.get_device()
    D = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
    mk = lambda *sh: D(torch.tensor(rng.standard_normal(sh), dtype=torch.float32))
    qv, kv, vv = mk(B, H, N, dh), mk(B, H, N, dh), mk(B, H, N, dh)
    bv = mk(1, H, N, N)
    seed = mk(B, H, N, dh)
    shim = tp.taped_ttnn()
    itemsize = 2
    cB, cQ = tp._sdpa_chunking(B, H, N, N, itemsize)

    def leaves():
        return [ag.Tensor(t, requires_grad=True) for t in (qv, kv, vv, bv)]

    def shipped_step():
        q, k, v, b = leaves()
        with ag.tape():
            o = shim.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=b, is_causal=False, scale=scale)
        o.backward(seed=seed)

    def twin_step():
        q, k, v, b = leaves()
        # The tape's own attention: the scores are materialised per chunk by matmul and
        # softmax rather than produced by the fused kernel. Same backward, by construction.
        o = ag.triangle_attention(q, k, v, b, scale=scale, chunk=cB, q_chunk=cQ)
        o.backward(seed=seed)

    arms = {"SHIPPED": shipped_step, "TWIN": twin_step}
    for fn in arms.values():
        for _ in range(a.warmup):
            fn()
    ttnn.synchronize_device(dev)
    stop, samples = threading.Event(), []
    th = threading.Thread(target=clocks, args=(stop, samples), daemon=True)
    th.start()
    times = {nm: [] for nm in arms}
    try:
        for _ in range(a.reps):
            for nm, fn in arms.items():
                t0 = time.perf_counter()
                fn()
                ttnn.synchronize_device(dev)
                times[nm].append(time.perf_counter() - t0)
    finally:
        stop.set(); th.join(timeout=3)
    print(f"triangle attention forward+backward, q/k/v [{B},{H},{N},{dh}] bias [1,{H},{N},{N}], "
          f"backward chunk {cB}/{cQ}, {a.reps} interleaved reps after {a.warmup} warmups")
    med = {}
    for nm in arms:
        v = sorted(times[nm])
        med[nm] = statistics.median(v)
        print(f"  {nm:8s} median {med[nm]*1e3:8.2f} ms   min {v[0]*1e3:8.2f}   "
              f"max {v[-1]*1e3:8.2f}   tokens/s {N/med[nm]:9.1f}")
    sp = med["TWIN"] / med["SHIPPED"]
    print(f"  SHIPPED is {sp:.3f}x the twin "
          f"({'faster' if sp > 1 else 'SLOWER -- negative result'})")
    if samples:
        print(f"  aiclk DURING: min {min(samples)} max {max(samples)} MHz "
              f"({len(samples)} samples)")
    print(f"  host load average: {os.getloadavg()}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ab", choices=("ab", "inference"))
    ap.add_argument("--case", default="transition", choices=("transition", "attention"))
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    N, c, hidden = a.tokens, a.c, a.hidden
    rng = np.random.default_rng(0)
    sd = weights(rng, c, hidden)
    x_t = torch.tensor(rng.standard_normal((1, N, N, c)), dtype=torch.float32)
    seed_t = torch.tensor(rng.standard_normal((1, N, N, c)), dtype=torch.float32)

    if a.case == "attention":
        return attention_ab(a, ttnn, tt, ag)

    dev = tt.get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    D = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
    module = tt.Transition(sd, ckc)

    if a.mode == "inference":
        x = D(x_t)
        out = module(x)
        h = hashlib.sha256(ttnn.to_torch(out).to(torch.float32).numpy().tobytes()).hexdigest()
        print(f"tokens={N} c={c} hidden={hidden}")
        print(f"untaped shipped forward digest: {h}")
        return 0

    # Leaves, once: both arms differentiate the same parameters.
    wt = {k: ag.Tensor(D(v.t() if v.dim() == 2 else v), requires_grad=True)
          for k, v in sd.items()}
    for attr, key in (("norm_weight", "norm.weight"), ("norm_bias", "norm.bias"),
                      ("fc1_weight", "fc1.weight"), ("fc2_weight", "fc2.weight"),
                      ("fc3_weight", "fc3.weight")):
        setattr(module, attr, wt[key])

    def shipped_step():
        x = ag.Tensor(D(x_t), requires_grad=True)
        with ag.tape():
            out = module(x)
        out.backward(seed=D(seed_t))
        return x

    def twin_step():
        """The same maths from tape primitives: no chunking, no L1 routing, no fused silu."""
        x = ag.Tensor(D(x_t), requires_grad=True)
        xn = ag.layer_norm(x, wt["norm.weight"], wt["norm.bias"], eps=1e-5)
        x1 = ag.silu(ag.linear(xn, wt["fc1.weight"]))
        x2 = ag.linear(xn, wt["fc2.weight"])
        out = ag.linear(ag.mul(x1, x2), wt["fc3.weight"])
        out.backward(seed=D(seed_t))
        return x

    arms = {"SHIPPED": shipped_step, "TWIN": twin_step}
    for nm, fn in arms.items():                       # compile + allocator warmup, both arms
        for _ in range(a.warmup):
            fn()
    ttnn.synchronize_device(dev)

    stop, samples = threading.Event(), []
    th = threading.Thread(target=clocks, args=(stop, samples), daemon=True)
    th.start()
    times = {nm: [] for nm in arms}
    try:
        for i in range(a.reps):
            for nm, fn in arms.items():               # INTERLEAVED, one for one
                t0 = time.perf_counter()
                fn()
                ttnn.synchronize_device(dev)
                times[nm].append(time.perf_counter() - t0)
    finally:
        stop.set(); th.join(timeout=3)

    print(f"Transition forward+backward, pair track [1,{N},{N},{c}] hidden {hidden}, "
          f"{a.reps} interleaved reps after {a.warmup} warmups")
    med = {}
    for nm in arms:
        v = sorted(times[nm])
        med[nm] = statistics.median(v)
        print(f"  {nm:8s} median {med[nm]*1e3:8.2f} ms   min {v[0]*1e3:8.2f}   "
              f"max {v[-1]*1e3:8.2f}   tokens/s {N/med[nm]:9.1f}")
    sp = med["TWIN"] / med["SHIPPED"]
    print(f"  SHIPPED is {sp:.3f}x the twin "
          f"({'faster' if sp > 1 else 'SLOWER -- negative result'})")
    if samples:
        print(f"  aiclk DURING: min {min(samples)} max {max(samples)} MHz "
              f"({len(samples)} samples)")
    print(f"  host load average: {os.getloadavg()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
