#!/usr/bin/env python3
"""The `ipa_fused.layer` op chain on a TT card: grade it against the float64 numpy backend, then time it.

Everything stays fp32 on the card (HiFi4, fp32 accumulate) for the first build: the point-folded
logits carry a pw/2 |kp|^2 key bias of a few hundred against differences of order one, which is the
precision question NOTES "Pass 2" left open, and fp32 answers it before anything is cast down.

  run on a LEASED card only (the row's #DISPATCH must say card=any and the lease must name it):
  TT_VISIBLE_DEVICES=<card> python perf/bc4_ipa/ipa_fused_tt.py --params <npz> [--n 224] [--reps 50] --out x.json

Prints the per-output rel L2 against the numpy backend, the eager and traced ms per layer, the op
count of one layer (calls into ttnn), and the AICLK read in-run.
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipa_fused as F

CALLS = [0]


def counted(fn):
    def f(*a, **k):
        CALLS[0] += 1
        return fn(*a, **k)
    return f


class TT:
    """ttnn backend, fp32 TILE tensors in DRAM. Shapes match `ipa_fused.NP` op for op."""

    def __init__(self, dev):
        import ttnn
        self.t, self.dev = ttnn, dev
        self.ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                    fp32_dest_acc_en=True, packer_l1_acc=True)
        t = ttnn
        self.mm = counted(lambda a, b: t.matmul(a, b, compute_kernel_config=self.ckc))
        self.bmm = self.mm
        self.bmm_t = counted(lambda a, b: t.matmul(a, b, transpose_b=True, compute_kernel_config=self.ckc))
        self.add = counted(lambda a, b: t.add(a, b))
        self.sub = counted(lambda a, b: t.subtract(a, b))
        self.mul = counted(lambda a, b: t.multiply(a, b))
        self.relu = counted(t.relu)
        self.sqrt = counted(t.sqrt)
        self.recip = counted(t.reciprocal)
        self.concat = counted(lambda xs: t.concat(xs, dim=-1))
        self.permute = counted(lambda x, perm: t.permute(x, perm))

    def put(self, x):
        import torch
        return self.t.from_torch(torch.as_tensor(np.asarray(x, np.float32)), dtype=self.t.float32,
                                 layout=self.t.TILE_LAYOUT, device=self.dev)

    def get(self, x):
        return self.t.to_torch(x).float().numpy().astype(np.float64)

    def linear(self, x, w, b=None):
        CALLS[0] += 1
        return self.t.linear(x, w, bias=b, compute_kernel_config=self.ckc)

    def col(self, x, i):
        return self.cols(x, i, i + 1)

    def cols(self, x, s, e):
        CALLS[0] += 1
        sh = list(x.shape)
        return self.t.slice(x, [0] * (len(sh) - 1) + [s], sh[:-1] + [e])

    def heads(self, x, d):                    # [N, H*d] -> [H, N, d]
        CALLS[0] += 2
        n = int(x.shape[0])
        return self.t.permute(self.t.reshape(x, [n, F.H, d]), (1, 0, 2))

    def unheads(self, x):                     # [H, N, d] -> [N, H*d]
        CALLS[0] += 2
        h, n, d = (int(s) for s in x.shape)
        return self.t.reshape(self.t.permute(x, (1, 0, 2)), [n, h * d])

    def unsq(self, x):                        # [H, N] -> [H, 1, N]
        CALLS[0] += 1
        return self.t.reshape(x, [int(x.shape[0]), 1, int(x.shape[1])])

    def flat(self, x):                        # [N, H, C] -> [N, H*C]
        CALLS[0] += 1
        return self.t.reshape(x, [int(x.shape[0]), -1])

    def softmax(self, x):
        CALLS[0] += 1
        return self.t.softmax(x, dim=-1, numeric_stable=True, compute_kernel_config=self.ckc)

    def layer_norm(self, x, gb, eps=1e-5):
        CALLS[0] += 1
        return self.t.layer_norm(x, weight=gb[0], bias=gb[1], epsilon=eps, compute_kernel_config=self.ckc)


def upload(o, k):
    """Constants onto the card; 1-D biases and LN params as [1, w] rows, lists/tuples element-wise."""
    def up(v):
        if isinstance(v, (list, tuple)) and not isinstance(v[0], (int, np.integer)):
            return type(v)(up(x) for x in v)
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[0] == 3:
            return [up(x) for x in v]
        if isinstance(v, np.ndarray):
            return o.put(v[None] if v.ndim == 1 else v)
        return v
    return {name: (v if name == "kp_cols" else up(v)) for name, v in k.items()}


def aiclk():
    import subprocess
    try:
        return subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        return f"tt-smi failed: {e}"


def main(a):
    from tt_bio import tenstorrent as T
    raw = np.load(a.params)
    p = {kk: np.asarray(raw[kk], np.float64) for kk in raw.files if "/structure_module/" in kk}
    k = F.pack(p)
    rng = np.random.default_rng(1)
    n, real = a.n, a.n - 4
    act = rng.standard_normal((n, F.CS)); z = rng.standard_normal((n, n, F.CZ))
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    quat = rng.standard_normal((n, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = F.G.quaternion_rotation(quat[:, 1] / quat[:, 0], quat[:, 2] / quat[:, 0], quat[:, 3] / quat[:, 0]).reshape(n, 9)
    t = rng.standard_normal((n, 3)) * 5.0
    npo = F.NP()
    ref = F.layer(npo, k, act, R, t, F.pair_bias(npo, k, z, mask), z)

    dev = T.get_device(trace="diffusion")
    o = TT(dev)
    kd = upload(o, k)
    ins = [o.put(act), o.put(R), o.put(t), o.put(F.pair_bias(npo, k, z, mask)), o.put(z)]
    CALLS[0] = 0
    out = F.layer(o, kd, *ins)
    calls = CALLS[0]
    got = [o.get(x) for x in out]
    rel = {nm: float(np.linalg.norm(x - y) / np.linalg.norm(x))
           for nm, x, y in zip(("act", "rotation", "translation"), ref, got)}
    print("rel L2 vs float64:", rel, "ttnn calls per layer:", calls)

    import ttnn
    def timed(fn):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(a.reps):
            fn()
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / a.reps * 1e3
    eager = timed(lambda: F.layer(o, kd, *ins))
    tid, _ = T.capture_trace(dev, lambda: F.layer(o, kd, *ins))
    traced = timed(lambda: ttnn.execute_trace(dev, tid, cq_id=0, blocking=False))
    ttnn.release_trace(dev, tid)
    clk = aiclk()
    res = {"n": n, "rel_l2_vs_f64": rel, "ttnn_calls_per_layer": calls, "eager_ms_per_layer": eager,
           "traced_ms_per_layer": traced, "reps": a.reps, "dtype": "fp32 HiFi4",
           "visible": os.environ.get("TT_VISIBLE_DEVICES"), "tt_smi": clk[-4000:],
           "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps({kk: v for kk, v in res.items() if kk != "tt_smi"}))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out")
    main(ap.parse_args())
