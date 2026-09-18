#!/usr/bin/env python3
"""Does chunking let the attention backward run a size the retained route cannot?

An OOM boundary is a decisive measurement; a polled DRAM peak is not. `ttnn.get_memory_view`
behaves like a pipeline drain, and a poller thread competing for the GIL with a tight op
enqueue loop samples too rarely to catch a transient -- the first attempt at this reported the
output tensor as the peak and missed a 0.27 GB score tensor entirely. So this probes the
boundary instead: run each size with the scores held and with them chunked, and record which
one the card can actually complete.

It also checks the unfused forward against tt-bio's SHIPPED fused SDPA before quoting any
timing ratio between them, because a fused call that silently did less work would make the
unfused route look arbitrarily bad.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

HEADS, HEAD_DIM = 8, 32


def clocks_thread(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                               capture_output=True, text=True, timeout=25)
            for dev in json.loads(r.stdout).get("device_info", []):
                c = dev.get("telemetry", {}).get("aiclk")
                if c is not None:
                    out.append(int(c))
        except Exception:
            pass
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="256,512,800")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    stop, clocks = threading.Event(), []
    threading.Thread(target=clocks_thread, args=(stop, clocks), daemon=True).start()

    def free_gb():
        mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return mv.total_bytes_free_per_bank * mv.num_banks / 1e9

    def mk(shape):
        return ttnn.from_torch(torch.randn(*shape) * 0.5, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)

    print(f"# heads={HEADS} head_dim={HEAD_DIM}, chunk={args.chunk}. 'held' is the full")
    print(f"# [N,H,N,N] bf16 score tensor a retained-activation backward would need.")
    print(f"{'N':>5} {'held_GB':>8} {'free_GB':>8} {'HELD fwd+bwd':>14} {'CHUNKED fwd+bwd':>16} "
          f"{'chunk_fwd_s':>12} {'chunk_bwd_s':>12} {'bwd/fwd':>8}")

    for n in [int(x) for x in args.sizes.split(",")]:
        held = n * HEADS * n * n * 2 / 1e9
        q, k, v = (mk((n, HEADS, n, HEAD_DIM)) for _ in range(3))
        b, seed = mk((1, HEADS, n, n)), mk((n, HEADS, n, HEAD_DIM))
        avail = free_gb()
        results = {}
        for label, chunk in (("held", None), ("chunked", args.chunk)):
            try:
                tq, tk, tv, tb = (ag.Tensor(x, requires_grad=True) for x in (q, k, v, b))
                o = ag.triangle_attention(tq, tk, tv, tb, chunk=chunk, q_chunk=chunk)
                ttnn.synchronize_device(device)
                o.backward(seed=seed)
                ttnn.synchronize_device(device)
                results[label] = "OK"
                for obj in (tq, tk, tv, tb, o):
                    obj.grad, obj.node = None, None
                del tq, tk, tv, tb, o
            except Exception as e:
                results[label] = f"{type(e).__name__}"
        ft, wt = [], []
        if results.get("chunked") == "OK":
            for _ in range(args.reps):
                tq, tk, tv, tb = (ag.Tensor(x, requires_grad=True) for x in (q, k, v, b))
                t0 = time.perf_counter()
                o = ag.triangle_attention(tq, tk, tv, tb, chunk=args.chunk, q_chunk=args.chunk)
                ttnn.synchronize_device(device)
                t1 = time.perf_counter()
                o.backward(seed=seed)
                ttnn.synchronize_device(device)
                ft.append(t1 - t0)
                wt.append(time.perf_counter() - t1)
                for obj in (tq, tk, tv, tb, o):
                    obj.grad, obj.node = None, None
                del tq, tk, tv, tb, o
        f = statistics.median(ft) if ft else float("nan")
        w = statistics.median(wt) if wt else float("nan")
        print(f"{n:>5} {held:>8.2f} {avail:>8.2f} {results.get('held', '-'):>14} "
              f"{results.get('chunked', '-'):>16} {f:>12.3f} {w:>12.3f} {w / f:>8.2f}")
        for x in (q, k, v, b, seed):
            ttnn.deallocate(x)

    # --- fused vs unfused, agreement first then timing -----------------------
    n = 256
    q, k, v = (mk((n, HEADS, n, HEAD_DIM)) for _ in range(3))
    b = mk((1, HEADS, n, n))
    scale = HEAD_DIM ** -0.5
    print()
    # The fused kernel applies `scale` AFTER adding the bias -- compute_common.hpp computes
    # exp((qk + bias - max) * scale) -- so it wants the bias pre-multiplied by sqrt(d), which is
    # why tt-bio scales `bias_weight` by `self.scale` at construction (TriangleAttention.__init__,
    # `_bias_scale`). Passing the post-scale bias instead is silent: it shrinks the bias by
    # sqrt(32) = 5.66 and the first attempt at this comparison read cos 0.9187 and called the
    # forward wrong when the convention was wrong.
    b_fused = ttnn.multiply(b, 1.0 / scale)
    try:
        fused = tt._tri_att_sdpa(q, k, v, b_fused, scale)
        ttnn.synchronize_device(device)
        mine = ag.triangle_attention(ag.Tensor(q), ag.Tensor(k), ag.Tensor(v), ag.Tensor(b),
                                     scale=scale, chunk=args.chunk, q_chunk=args.chunk)
        fa = ttnn.to_torch(fused).to(torch.float64).numpy().ravel()
        ma = ttnn.to_torch(mine.value).to(torch.float64).numpy().ravel()
        print(f"# fused output shape {tuple(fused.shape)}, unfused {tuple(mine.value.shape)}")
        rel = np.linalg.norm(ma - fa) / np.linalg.norm(fa)
        cos = float(ma @ fa / (np.linalg.norm(ma) * np.linalg.norm(fa)))
        print(f"# unfused forward vs SHIPPED fused SDPA at N={n}: rel_l2 {rel:.2e}, cos {cos:.6f}")
        agrees = rel < 2e-2
        print(f"# {'agree -- the timing comparison below is between two real computations'if agrees else 'DISAGREE -- do not quote a timing ratio, one of them is not doing the work'}")
        if agrees:
            tf, tu = [], []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                o1 = tt._tri_att_sdpa(q, k, v, b_fused, scale)
                ttnn.synchronize_device(device)
                tf.append(time.perf_counter() - t0)
                ttnn.deallocate(o1)
                t0 = time.perf_counter()
                o2 = ag.triangle_attention(ag.Tensor(q), ag.Tensor(k), ag.Tensor(v),
                                           ag.Tensor(b), scale=scale,
                                           chunk=args.chunk, q_chunk=args.chunk)
                ttnn.synchronize_device(device)
                tu.append(time.perf_counter() - t0)
                ttnn.deallocate(o2.value)
            mf, mu = statistics.median(tf), statistics.median(tu)
            print(f"# fused fwd {mf * 1e3:.2f} ms, unfused chunked fwd {mu * 1e3:.2f} ms "
                  f"-> unfusing costs {mu / mf:.2f}x on the forward")
    except Exception as e:
        print(f"# fused comparison failed: {type(e).__name__}: {str(e)[:120]}")

    stop.set()
    time.sleep(0.1)
    if clocks:
        print(f"\nAICLK during: min {min(clocks)} max {max(clocks)} "
              f"median {int(statistics.median(clocks))} MHz over {len(clocks)} samples")
    else:
        print("\nAICLK: NO SAMPLES -- every timing above is unclocked and unusable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
