#!/usr/bin/env python3
"""The `ipa_fused.layer` op chain on a TT card: grade it against the float64 numpy backend, then time it.

Everything stays fp32 on the card (HiFi4, fp32 accumulate) for the first build: the point-folded
logits carry a pw/2 |kp|^2 key bias of a few hundred against differences of order one, which is the
precision question NOTES "Pass 2" left open, and fp32 answers it before anything is cast down.

  run on a LEASED card only (the row's #DISPATCH must say card=any and the lease must name it):
  TT_VISIBLE_DEVICES=<card> python perf/bc4_ipa/ipa_fused_tt.py --params <npz> [--n 224] [--reps 50] [--grad] --out x.json
  python perf/bc4_ipa/ipa_fused_tt.py --params <npz> --jax-check     (card-free: the JAX reference vs numpy and finite differences)

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
        self.sqrt_eps = counted(lambda x, eps: t.sqrt(t.add(x, eps)))
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

    def flat(self, x):                        # [N, H, C] -> [N, H*C]
        CALLS[0] += 1
        return self.t.reshape(x, [int(x.shape[0]), -1])

    def softmax(self, x):
        # the logits are exact (small); ttnn's fp32 row max is not trusted at |x| ~ 1e3 (NOTES "Pass 5")
        CALLS[0] += 1
        return self.t.softmax(x, dim=-1, compute_kernel_config=self.ckc)

    def layer_norm(self, x, gb, eps=1e-5):
        CALLS[0] += 1
        return self.t.layer_norm(x, weight=gb[0], bias=gb[1], epsilon=eps, compute_kernel_config=self.ckc)


class AG:
    """tt_bio.autograd backend: the same chain on the tape, so the backward is the tape's VJPs.
    Weights and the pair bias are constants (BindCraft 2 differentiates the sequence, not the
    weights); act, R and t carry gradients."""

    def __init__(self, dev):
        import ttnn
        from tt_bio import autograd as A
        self.t, self.A, self.dev = ttnn, A, dev
        self.inner = TT(dev)

    def put(self, x, grad=False):
        return self.A.Tensor(self.inner.put(x), requires_grad=grad)

    def get(self, x):
        return self.inner.get(x.value if hasattr(x, "value") else x)

    def mm(self, a, b): CALLS[0] += 1; return self.A.matmul(a, b)
    bmm = mm
    def bmm_t(self, a, b): CALLS[0] += 1; return self.A.matmul(a, b, transpose_b=True)
    def linear(self, x, w, b=None): CALLS[0] += 1; return self.A.linear(x, w, b)
    def add(self, a, b): CALLS[0] += 1; return self.A.add(a, b)
    def mul(self, a, b): CALLS[0] += 1; return self.A.mul(a, b)
    def sub(self, a, b): CALLS[0] += 2; return self.A.add(a, self.A.scale(b, -1.0))
    def cols(self, x, s, e): CALLS[0] += 1; return self.A.narrow(x, -1, s, e - s)
    def col(self, x, i): return self.cols(x, i, i + 1)
    def concat(self, xs): CALLS[0] += 1; return self.A.concat(xs, dim=-1)
    def permute(self, x, perm): CALLS[0] += 1; return self.A.permute(x, perm)
    def heads(self, x, d):
        CALLS[0] += 2
        return self.A.permute(self.A.reshape(x, [int(x.value.shape[0]), F.H, d]), (1, 0, 2))
    def unheads(self, x):
        CALLS[0] += 2
        h, n, d = (int(s) for s in x.value.shape)
        return self.A.reshape(self.A.permute(x, (1, 0, 2)), [n, h * d])
    def flat(self, x): CALLS[0] += 1; return self.A.reshape(x, [int(x.value.shape[0]), -1])
    def relu(self, x): CALLS[0] += 1; return self.A.relu(x)
    def softmax(self, x): CALLS[0] += 1; return self.A.softmax(x, -1, config=self.inner.ckc)
    def layer_norm(self, x, gb, eps=1e-5): CALLS[0] += 1; return self.A.layer_norm(x, gb[0], gb[1], eps=eps)

    def sqrt_eps(self, x, eps):
        """sqrt(x + eps); backward g / (2 out)."""
        CALLS[0] += 2
        t, out_v = self.t, self.t.sqrt(self.t.add(x.value, eps))
        def make():
            def bw(g):
                x.add_grad(t.multiply(g, t.reciprocal(t.multiply(out_v, 2.0))))
            return bw
        return self.A._tape(out_v, [x], make)

    def recip(self, x):
        """1 / x; backward -g out^2."""
        CALLS[0] += 1
        t, out_v = self.t, self.t.reciprocal(x.value)
        def make():
            def bw(g):
                x.add_grad(t.multiply(g, t.multiply(t.multiply(out_v, out_v), -1.0)))
            return bw
        return self.A._tape(out_v, [x], make)


def jax_reference(k, act, R, t, bias, z, seeds):
    """float64 VJP of the same chain in JAX: the backward's reference, host only."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    class J(F.NP):
        put = staticmethod(jnp.asarray)
        concat = staticmethod(lambda xs: jnp.concatenate(xs, -1))
        heads = staticmethod(lambda x, d: x.reshape(x.shape[0], F.H, d).transpose(1, 0, 2))
        unheads = staticmethod(lambda x: x.transpose(1, 0, 2).reshape(x.shape[1], -1))
        bmm_t = staticmethod(lambda a, b: a @ jnp.swapaxes(b, -1, -2))
        relu = staticmethod(lambda x: jnp.maximum(x, 0))
        sqrt_eps = staticmethod(lambda x, eps: jnp.sqrt(x + eps))
        add = staticmethod(jnp.add); sub = staticmethod(jnp.subtract); mul = staticmethod(jnp.multiply)
        def softmax(self, x): return jax.nn.softmax(x, -1)
        def layer_norm(self, x, gb, eps=1e-5):
            mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) / jnp.sqrt(var + eps) * gb[0] + gb[1]
    o = J()
    f = lambda a_, r_, t_: F.layer(o, k, a_, r_, t_, jnp.asarray(bias), jnp.asarray(z))
    out, vjp = jax.vjp(f, jnp.asarray(act), jnp.asarray(R), jnp.asarray(t))
    return [np.asarray(x) for x in out], [np.asarray(g) for g in vjp(tuple(jnp.asarray(s) for s in seeds))]


def grad_check(dev, k, act, R, t, bias, z, reps):
    """VJP of one layer through the tape, against `jax_reference`; seeds are fixed random cotangents."""
    import ttnn
    from tt_bio import autograd as A
    rng = np.random.default_rng(7)
    ref_out, ref_g = None, None
    seeds = [rng.standard_normal(x.shape) for x in (act, R, t)]
    ref_out, ref_g = jax_reference(k, act, R, t, bias, z, seeds)
    o = AG(dev)
    kd = upload(o, k)

    def step():
        ins = [o.put(act, True), o.put(R, True), o.put(t, True)]
        out = F.layer(o, kd, *ins, o.put(bias), o.put(z))
        A.backward(list(out), seeds=[o.inner.put(s_) for s_ in seeds])
        return ins
    CALLS[0] = 0
    ins = step()
    fwd_calls = CALLS[0]
    got = [o.get(x.grad) for x in ins]
    rel = {nm: float(np.linalg.norm(g - r) / np.linalg.norm(r)) for nm, g, r in zip(("d_act", "d_R", "d_t"), got, ref_g)}
    ttnn.synchronize_device(dev); t0 = time.perf_counter()
    for _ in range(max(3, reps // 10)):
        step()
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) / max(3, reps // 10) * 1e3
    print("backward rel L2 vs JAX float64:", rel, "eager fwd+bwd ms", ms, flush=True)
    return {"rel_l2_vs_jax_f64": rel, "eager_fwd_bwd_ms_per_layer": ms, "taped_forward_calls": fwd_calls}


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


class Clock:
    """AICLK of every chip tt-smi lists, sampled every ~2 s while the `with` body runs (in-run, not before)."""
    def __enter__(self):
        import threading
        self.samples, self.stop = [], threading.Event()
        def loop():
            while not self.stop.is_set():
                try:
                    d = json.loads(aiclk())
                    self.samples.append([int(x["telemetry"]["aiclk"]) for x in d["device_info"]])
                except Exception:
                    pass
                self.stop.wait(2.0)
        self.th = threading.Thread(target=loop, daemon=True); self.th.start()
        return self

    def __exit__(self, *e):
        self.stop.set(); self.th.join()

    def median(self, chip):
        v = sorted(x[chip] for x in self.samples if len(x) > chip)
        return v[len(v) // 2] if v else None


def jax_check(a):
    raw = np.load(a.params)
    p = {kk: np.asarray(raw[kk], np.float64) for kk in raw.files if "/structure_module/" in kk}
    k = F.pack(p)
    rng = np.random.default_rng(1)
    n = 64
    act = rng.standard_normal((n, F.CS)); z = rng.standard_normal((n, n, F.CZ)); mask = np.ones(n)
    quat = rng.standard_normal((n, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = F.G.quaternion_rotation(quat[:, 1] / quat[:, 0], quat[:, 2] / quat[:, 0], quat[:, 3] / quat[:, 0]).reshape(n, 9)
    t = rng.standard_normal((n, 3)) * 5.0
    o = F.NP(); bias = F.pair_bias(o, k, z, mask)
    seeds = [rng.standard_normal(x.shape) for x in (act, R, t)]
    out, g = jax_reference(k, act, R, t, bias, z, seeds)
    ref = F.layer(o, k, act, R, t, bias, z)
    print("jax forward vs numpy:", max(float(np.abs(x - y).max()) for x, y in zip(out, ref)))
    loss = lambda a_, r_, t_: sum(float((x * s_).sum()) for x, s_ in zip(F.layer(o, k, a_, r_, t_, bias, z), seeds))
    for nm, i, x in (("act", 0, act), ("R", 1, R), ("t", 2, t)):
        d = rng.standard_normal(x.shape); h = 1e-6
        args = [act, R, t]
        up = list(args); up[i] = x + h * d; dn = list(args); dn[i] = x - h * d
        fd = (loss(*up) - loss(*dn)) / (2 * h)
        print(f"d_{nm}: vjp.d {float((g[i] * d).sum()):.8e}  finite difference {fd:.8e}")


def main(a):
    if a.jax_check:
        return jax_check(a)
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
    clock = Clock().__enter__()
    eager = timed(lambda: F.layer(o, kd, *ins))
    tid, _ = T.capture_trace(dev, lambda: F.layer(o, kd, *ins))
    traced = timed(lambda: ttnn.execute_trace(dev, tid, cq_id=0, blocking=False))
    ttnn.release_trace(dev, tid)
    if a.grad:
        res_grad = grad_check(dev, k, act, R, t, F.pair_bias(npo, k, z, mask), z, a.reps)
    clock.__exit__()
    clk = aiclk()
    res = {"aiclk_samples": clock.samples, "aiclk_median_by_chip": [clock.median(i) for i in range(len(clock.samples[0]))] if clock.samples else None,
           "loadavg": os.getloadavg(), "n": n, "rel_l2_vs_f64": rel, "ttnn_calls_per_layer": calls, "eager_ms_per_layer": eager,
           "traced_ms_per_layer": traced, "reps": a.reps, "dtype": "fp32 HiFi4",
           "visible": os.environ.get("TT_VISIBLE_DEVICES"), "tt_smi": clk[-4000:],
           "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.grad:
        res["backward"] = res_grad
    print(json.dumps({kk: v for kk, v in res.items() if kk != "tt_smi"}))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out")
    ap.add_argument("--jax-check", action="store_true")
    ap.add_argument("--grad", action="store_true", help="also tape the layer and grade its VJP against JAX float64")
    main(ap.parse_args())
