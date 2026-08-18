# Direct sm_100 smoke test of the two cuEquivariance triangle primitives at the shapes a
# 512-aa fold uses. This is where both models hang, so it is the cheapest possible probe:
# a hang shows up in 60 s instead of costing a 20-minute fold timeout.
import os, sys, time, torch
import cuequivariance_ops_torch as c
import cuequivariance_ops as co
dev = "cuda"; N = 512; H = 4; D = 32
print("torch", torch.__version__, "cueq_ops", getattr(co, "__version__", "?"),
      "cap", torch.cuda.get_device_capability(), flush=True)

def timed(name, fn):
    torch.cuda.synchronize(); t0 = time.time()
    out = fn()
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"{name}: OK {dt*1000:.1f} ms  out {tuple(out.shape)} {out.dtype} "
          f"finite={bool(torch.isfinite(out).all())} absmax={out.abs().max().item():.4f}", flush=True)
    return out

torch.manual_seed(0)
# triangle_attention: (B, N, H, N, D) is the pair-stack layout protenix feeds it
q = torch.randn(1, N, H, N, D, device=dev, dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
bias = torch.randn(1, 1, H, N, N, device=dev, dtype=torch.bfloat16)
mask = torch.ones(1, N, 1, 1, N, device=dev, dtype=torch.bool)
timed("triangle_attention", lambda: c.triangle_attention(q, k, v, bias, mask))

# triangle_multiplicative_update: (B, N, N, C)
C = 128
x = torch.randn(1, N, N, C, device=dev, dtype=torch.bfloat16)
m = torch.ones(1, N, N, device=dev, dtype=torch.bfloat16)
for direction in ("outgoing", "incoming"):
    timed(f"triangle_multiplicative_update[{direction}]",
          lambda d=direction: c.triangle_multiplicative_update(x, direction=d, mask=m))
print("SMOKE ALL OK", flush=True)
