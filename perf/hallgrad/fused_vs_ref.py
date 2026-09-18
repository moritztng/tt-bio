#!/usr/bin/env python3
"""Fused SDPA and the unfused chunked route, each against a float64 reference.

Comparing the two against each other cannot say which is right, only that they differ. Both
are approximations, so the reference has to be float64 on the same rounded inputs.
"""
import json
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

HEADS, HEAD_DIM, N = 8, 32, 256
CHUNK = 32


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
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    stop, clocks = threading.Event(), []
    threading.Thread(target=clocks_thread, args=(stop, clocks), daemon=True).start()
    scale = HEAD_DIM ** -0.5
    rng = np.random.default_rng(11)

    # round to bf16 first, then the reference sees exactly what the device saw
    host = {k: torch.from_numpy(rng.standard_normal(s) * 0.5).to(torch.bfloat16)
            for k, s in (("q", (N, HEADS, N, HEAD_DIM)), ("k", (N, HEADS, N, HEAD_DIM)),
                         ("v", (N, HEADS, N, HEAD_DIM)), ("b", (1, HEADS, N, N)))}
    ref64 = {k: v.to(torch.float64) for k, v in host.items()}
    s = ref64["q"] @ ref64["k"].transpose(-2, -1) * scale + ref64["b"]
    ref = (torch.softmax(s, dim=-1) @ ref64["v"]).numpy().ravel()
    del s

    dev = {k: ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
           for k, v in host.items()}
    b_fused = ttnn.multiply(dev["b"], 1.0 / scale)  # fused kernel scales AFTER adding bias

    def rel(a):
        a = a.ravel()
        return (float(np.linalg.norm(a - ref) / np.linalg.norm(ref)),
                float(a @ ref / (np.linalg.norm(a) * np.linalg.norm(ref))))

    fused = tt._tri_att_sdpa(dev["q"], dev["k"], dev["v"], b_fused, scale)
    ttnn.synchronize_device(device)
    mine = ag.triangle_attention(ag.Tensor(dev["q"]), ag.Tensor(dev["k"]), ag.Tensor(dev["v"]),
                                 ag.Tensor(dev["b"]), scale=scale, chunk=CHUNK, q_chunk=CHUNK)
    fr, fc = rel(ttnn.to_torch(fused).to(torch.float64).numpy())
    ur, uc = rel(ttnn.to_torch(mine.value).to(torch.float64).numpy())
    print(f"# N={N} heads={HEADS} head_dim={HEAD_DIM}, both vs the SAME float64 reference")
    print(f"{'route':<34} {'rel_l2':>10} {'cos':>10}")
    print(f"{'shipped fused SDPA (op-default cfg)':<34} {fr:>10.2e} {fc:>10.6f}")
    print(f"{'unfused chunked (HiFi4/fp32_acc)':<34} {ur:>10.2e} {uc:>10.6f}")
    print(f"# ratio of errors: unfused is {fr / ur:.2f}x closer to float64 than the fused kernel"
          if ur < fr else
          f"# ratio of errors: fused is {ur / fr:.2f}x closer to float64 than the unfused route")
    ttnn.deallocate(fused)
    ttnn.deallocate(mine.value)

    tf, tu = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        o = tt._tri_att_sdpa(dev["q"], dev["k"], dev["v"], b_fused, scale)
        ttnn.synchronize_device(device)
        tf.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
        t0 = time.perf_counter()
        o = ag.triangle_attention(ag.Tensor(dev["q"]), ag.Tensor(dev["k"]), ag.Tensor(dev["v"]),
                                  ag.Tensor(dev["b"]), scale=scale, chunk=CHUNK, q_chunk=CHUNK)
        ttnn.synchronize_device(device)
        tu.append(time.perf_counter() - t0)
        ttnn.deallocate(o.value)
    mf, mu = statistics.median(tf), statistics.median(tu)
    print(f"\n# forward, median of 5: fused {mf * 1e3:.2f} ms, unfused chunked {mu * 1e3:.2f} ms")
    print(f"# unfusing the triangle attention forward costs {mu / mf:.1f}x at N={N}")
    stop.set()
    time.sleep(0.1)
    if clocks:
        print(f"# AICLK during: min {min(clocks)} max {max(clocks)} "
              f"median {int(statistics.median(clocks))} MHz over {len(clocks)} samples")
    else:
        print("# AICLK: NO SAMPLES -- timings unusable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
