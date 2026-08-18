# CUDA_LAUNCH_BLOCKING=1 makes every launch synchronous, so the faulthandler frame lands on
# the op that actually hangs rather than on the first downstream synchronising call. Without
# this, an async triton kernel that never completes shows up as a hang in torch.einsum.
import faulthandler, torch, time
import cuequivariance_ops_torch as c
import cuequivariance_ops as co
print("cueq_ops", getattr(co, "__version__", "?"), "torch", torch.__version__, flush=True)

_orig = torch.einsum
def logged(eq, *ops, **kw):
    shapes = [tuple(o.shape) for o in ops if hasattr(o, "shape")]
    dts = [str(o.dtype) for o in ops if hasattr(o, "dtype")]
    print(f"EINSUM {eq} shapes={shapes} dtypes={dts}", flush=True)
    return _orig(eq, *ops, **kw)
torch.einsum = logged
torch.functional.einsum = logged

N, C = 512, 128
x = torch.randn(1, N, N, C, device="cuda", dtype=torch.bfloat16)
m = torch.ones(1, N, N, device="cuda", dtype=torch.bfloat16)
faulthandler.dump_traceback_later(60, exit=True)
print("calling trimul outgoing", flush=True)
t0 = time.time()
out = c.triangle_multiplicative_update(x, direction="outgoing", mask=m)
torch.cuda.synchronize()
print("trimul OK", (time.time()-t0)*1000, "ms", flush=True)
