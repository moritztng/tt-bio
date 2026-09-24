#!/usr/bin/env python3
"""Taped TriangleAttention forward + backward at n tokens, timed, on whatever tree is on PYTHONPATH.

The construction is taped_triatt.py's (fp32_softmax, scale_pair_bias=False, no linear biases),
which is AF2's and OpenFold3's. Under a tape the fused-HiFi route declines, so every rep runs
`_fp32_softmax_attention`, the site e41512570 changes. Two warm reps, then timed reps, each
bracketed by synchronize_device. Prints wall, process CPU, AICLK from sysfs during the timed
reps, the fp32-softmax block stats, and a sha256 of dL/dz so the two trees can be compared.
"""
import argparse, hashlib, json, math, os, threading, time
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sysfs", default="/sys/class/tenstorrent/tenstorrent!0")
    a = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    rng = np.random.default_rng(0)
    S, c_z, heads, hd = a.tokens, 128, 4, 32
    t = lambda *sh: torch.tensor(rng.standard_normal(sh) / math.sqrt(sh[-1]), dtype=torch.float32)
    sd = {"layer_norm.weight": torch.tensor(rng.standard_normal(c_z) * 0.2 + 1.0, dtype=torch.float32),
          "layer_norm.bias": torch.tensor(rng.standard_normal(c_z) * 0.05, dtype=torch.float32),
          "linear_q.weight": t(heads * hd, c_z), "linear_k.weight": t(heads * hd, c_z),
          "linear_v.weight": t(heads * hd, c_z), "linear_g.weight": t(heads * hd, c_z),
          "linear_o.weight": t(c_z, heads * hd), "linear.weight": t(heads, c_z)}
    dev = tt.get_device()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    mod = tt.TriangleAttention(hd, heads, False, sd, ckc, scale_pair_bias=False,
                               fp32_softmax=True, fused_hifi=True)
    D = lambda x: ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
    x0 = torch.tensor(rng.standard_normal((1, S, S, c_z)), dtype=torch.float32)
    lw = torch.tensor(rng.standard_normal((1, S, S, c_z)), dtype=torch.float32)
    xd, sd_ = D(x0), D(lw)

    def step():
        xa = ag.Tensor(xd, requires_grad=True)
        with ag.tape():
            out = mod(xa)
        out.backward(seed=sd_)
        return xa.grad

    for _ in range(2):
        g = step()
    ttnn.synchronize_device(dev)
    clk, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            try:
                clk.append(int(open(os.path.join(a.sysfs, "tt_aiclk")).read().strip()))
            except Exception:
                pass
            time.sleep(0.25)
    th = threading.Thread(target=sample, daemon=True); th.start()
    walls, cpus = [], []
    for _ in range(a.reps):
        ttnn.synchronize_device(dev)
        w0, c0 = time.perf_counter(), time.process_time()
        g = step()
        ttnn.synchronize_device(dev)
        walls.append(time.perf_counter() - w0); cpus.append(time.process_time() - c0)
    stop.set(); th.join()
    gt = ttnn.to_torch(g).float()
    rep = {"tag": a.tag, "tokens": S, "walls": walls, "cpus": cpus,
           "wall_median": float(np.median(walls)), "cpu_median": float(np.median(cpus)),
           "aiclk_during": [min(clk), max(clk), len(clk)] if clk else None,
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "fp32_softmax_stats": {k: v for k, v in tt.FP32_SOFTMAX_STATS.items() if isinstance(v, (int, float))},
           "grad_sha": hashlib.sha256(gt.numpy().tobytes()).hexdigest()[:16],
           "grad_norm": float(gt.double().norm())}
    print("RESULT " + json.dumps(rep))


if __name__ == "__main__":
    main()
