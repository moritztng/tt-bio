#!/usr/bin/env python3
"""What the chunked triangle-attention backward costs in DRAM and in wall clock.

Two questions the build has to answer with numbers rather than argument:
  1. Does chunking keep the score tensor out of DRAM, and by how much?
  2. What does the unfused route cost against the fused SDPA the forward ships?

Timing and peak DRAM are measured in SEPARATE passes on purpose. `ttnn.get_memory_view`
behaves like a pipeline drain (tenstorrent.py:5073), so polling it inside a timed region
inflates the time it is meant to measure. So: timed reps with no polling, then one untimed
rep with a 5 ms DRAM poller for the peak. Every timing carries the AICLK sampled DURING it.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

import torch

HEADS, HEAD_DIM = 8, 32


def clock_sampler(stop, out):
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
    ap.add_argument("--sizes", default="256,512")
    ap.add_argument("--chunks", default="32,128,none")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fused", action="store_true")
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    stop, clocks = threading.Event(), []
    threading.Thread(target=clock_sampler, args=(stop, clocks), daemon=True).start()

    def dram_gb():
        mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return mv.total_bytes_allocated_per_bank * mv.num_banks / 1e9

    class Peak:
        """Poll DRAM at 5 ms and keep the max. Untimed regions only."""

        def __enter__(self):
            self.stop, self.peak = threading.Event(), 0.0
            self.base = dram_gb()

            def run():
                while not self.stop.is_set():
                    try:
                        self.peak = max(self.peak, dram_gb() - self.base)
                    except Exception:
                        pass
                    time.sleep(0.005)
            self.t = threading.Thread(target=run, daemon=True)
            self.t.start()
            return self

        def __exit__(self, *exc):
            self.stop.set()
            self.t.join(timeout=2)
            return False

    def mk(shape):
        return ttnn.from_torch(torch.randn(*shape) * 0.5, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)

    print(f"# heads={HEADS} head_dim={HEAD_DIM}. 'held' is the full [N,H,N,N] bf16 score tensor")
    print(f"# a retained-activation backward would need. Peaks are over q/k/v/bias already resident.")
    print(f"{'N':>5} {'chunk':>6} {'held_GB':>9} {'fwd_pk_GB':>10} {'bwd_pk_GB':>10} "
          f"{'vs_held':>8} {'fwd_s':>8} {'bwd_s':>8} {'bwd/fwd':>8}")
    for n in [int(x) for x in args.sizes.split(",")]:
        held = n * HEADS * n * n * 2 / 1e9
        q, k, v = mk((n, HEADS, n, HEAD_DIM)), mk((n, HEADS, n, HEAD_DIM)), mk((n, HEADS, n, HEAD_DIM))
        b, seed = mk((1, HEADS, n, n)), mk((n, HEADS, n, HEAD_DIM))
        for cs in args.chunks.split(","):
            chunk = None if cs == "none" else int(cs)

            def one(measure_peak):
                tq, tk, tv, tb = (ag.Tensor(x, requires_grad=True) for x in (q, k, v, b))
                t0 = time.perf_counter()
                o = ag.triangle_attention(tq, tk, tv, tb, chunk=chunk, q_chunk=chunk)
                ttnn.synchronize_device(device)
                t1 = time.perf_counter()
                fp = measure_peak() if measure_peak else 0.0
                o.backward(seed=seed)
                ttnn.synchronize_device(device)
                t2 = time.perf_counter()
                for obj in (tq, tk, tv, tb, o):
                    obj.grad, obj.node = None, None
                return t1 - t0, t2 - t1, fp
            try:
                # pass 1: timing only, no DRAM polling
                times = [one(None)[:2] for _ in range(args.reps)]
                f = statistics.median([x[0] for x in times])
                w = statistics.median([x[1] for x in times])
                # pass 2: untimed, polled for the peak
                with Peak() as pf:
                    tq, tk, tv, tb = (ag.Tensor(x, requires_grad=True) for x in (q, k, v, b))
                    o = ag.triangle_attention(tq, tk, tv, tb, chunk=chunk, q_chunk=chunk)
                    ttnn.synchronize_device(device)
                fwd_pk = pf.peak
                with Peak() as pb:
                    o.backward(seed=seed)
                    ttnn.synchronize_device(device)
                bwd_pk = pb.peak
                for obj in (tq, tk, tv, tb, o):
                    obj.grad, obj.node = None, None
                del tq, tk, tv, tb, o
                worst = max(fwd_pk, bwd_pk)
                print(f"{n:>5} {cs:>6} {held:>9.2f} {fwd_pk:>10.3f} {bwd_pk:>10.3f} "
                      f"{held / worst if worst else float('nan'):>7.1f}x {f:>8.3f} {w:>8.3f} "
                      f"{w / f:>8.2f}")
            except Exception as e:
                print(f"{n:>5} {cs:>6} {held:>9.2f}  FAILED: {type(e).__name__}: {str(e)[:70]}")
        if args.fused:
            try:
                ft = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    o = tt._tri_att_sdpa(q, k, v, b, HEAD_DIM ** -0.5)
                    ttnn.synchronize_device(device)
                    ft.append(time.perf_counter() - t0)
                    ttnn.deallocate(o)
                print(f"{n:>5} {'FUSED':>6} {'':>9} {'':>10} {'':>10} {'':>8} "
                      f"{statistics.median(ft):>8.3f}  <- shipped fused SDPA forward")
            except Exception as e:
                print(f"{n:>5} {'FUSED':>6}  probe failed: {type(e).__name__}: {str(e)[:70]}")
        for x in (q, k, v, b, seed):
            ttnn.deallocate(x)

    stop.set()
    time.sleep(0.1)
    if clocks:
        print(f"\nAICLK during: min {min(clocks)} max {max(clocks)} "
              f"median {int(statistics.median(clocks))} MHz over {len(clocks)} samples")
    else:
        print("\nAICLK: NO SAMPLES -- treat every timing above as unclocked and unusable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
