"""Host AdamW elementwise: optim.py's numpy op order vs the same ops as in-place torch on the same buffers (threaded). Bitwise + time."""
import time, numpy as np, torch, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 178_448_304
rng = np.random.default_rng(0); f = np.float32
th0 = rng.standard_normal(N, dtype=f) * f(0.02)
b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
def numpy_step(th, m, v, g, lr, bc1, bc2):
    m *= b1; m += (1.0 - b1) * g; v *= b2; v += (1.0 - b2) * (g * g)
    upd = lr * ((m / bc1) / (np.sqrt(v / bc2) + eps) + wd * th); th -= upd; return upd
def torch_step(th, m, v, g, lr, bc1, bc2):
    T = torch.from_numpy; th, m, v, g = T(th), T(m), T(v), T(g)
    m.mul_(b1).add_(g.mul(1.0 - b1)); v.mul_(b2).add_(g.mul(g).mul_(1.0 - b2))
    upd = m.div(bc1).div_(v.div(bc2).sqrt_().add_(eps)).add_(th.mul(wd)).mul_(lr); th.sub_(upd); return upd.numpy()
S = {}
for name, fn in (("numpy", numpy_step), ("torch", torch_step)):
    th, m, v = th0.copy(), np.zeros(N, f), np.zeros(N, f); ts = []
    for k in range(1, 4):
        g = np.random.default_rng(k).standard_normal(N, dtype=f) * f(1e-3)
        t0 = time.perf_counter(); fn(th, m, v, g, 1e-4, 1 - b1 ** k, 1 - b2 ** k); ts.append(time.perf_counter() - t0)
    S[name] = (th, m, v); print(name, "s/step", [round(t, 3) for t in ts], flush=True)
for i, nm in enumerate(("theta", "m", "v")):
    a, b = S["numpy"][i], S["torch"][i]
    print(nm, "bit-identical fraction %.9f" % np.mean(a.view(np.int32) == b.view(np.int32)), "max rel %.2e" % (np.abs(a - b).max() / np.abs(a).max()))
print("torch threads", torch.get_num_threads(), "numpy", np.__version__, "torch", torch.__version__)
