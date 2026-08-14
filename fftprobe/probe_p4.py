#!/usr/bin/env python3
"""Is ANY fp32 data movement exact on Blackhole? The FFT butterfly needs an intra-tile shuffle.
p2 showed fp32 eltwise is bit-exact but ttnn.transpose is not (rel 4.15e-4). If a matmul by the
identity is also lossy, every FPU-mediated shuffle costs ~11-bit rounding, and the only exact
intra-tile movement left is a row-major round trip -- which is exactly the rebank_rm path
PR #44030 built and never benchmarked."""
import json
import numpy as np, torch, ttnn
R = {}
dev = ttnn.open_device(device_id=0)
def cfg(fid, acc=True):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid),
                                            math_approx_mode=False, fp32_dest_acc_en=acc, packer_l1_acc=False)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
torch.manual_seed(0)
x = torch.randn(1, 1, 256, 32, dtype=torch.float32)
I = torch.eye(32, dtype=torch.float32).reshape(1, 1, 32, 32)
P = torch.eye(32, dtype=torch.float32)[torch.tensor([(i + 16) % 32 for i in range(32)])].reshape(1, 1, 32, 32)
tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
R["identity_matmul"] = {}; R["perm_matmul"] = {}
for fid in ("LoFi", "HiFi2", "HiFi3", "HiFi4"):
    for nm, M, ref in (("identity_matmul", I, x), ("perm_matmul", P, x @ P.reshape(32, 32))):
        tm = ttnn.from_torch(M, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        g = ttnn.to_torch(ttnn.matmul(tx, tm, compute_kernel_config=cfg(fid), dtype=ttnn.float32))
        R[nm][fid] = {"bit_exact": bool(torch.equal(g, ref)), "rel": rel(g.double().numpy(), ref.double().numpy())}
print("identity matmul", json.dumps(R["identity_matmul"], indent=1), flush=True)
print("perm matmul", json.dumps(R["perm_matmul"], indent=1), flush=True)

# row-major round trip (untilize -> tilize), the only movement path PR #44030 trusted
rm = ttnn.to_layout(tx, ttnn.ROW_MAJOR_LAYOUT)
back = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
R["rowmajor_roundtrip_bit_exact"] = bool(torch.equal(ttnn.to_torch(back), x))
print("rowmajor roundtrip bit exact:", R["rowmajor_roundtrip_bit_exact"], flush=True)
# transpose on a row-major tensor
try:
    tr = ttnn.transpose(rm, -2, -1)
    g = ttnn.to_torch(ttnn.to_layout(tr, ttnn.TILE_LAYOUT) if tr.layout != ttnn.TILE_LAYOUT else tr)
    R["rowmajor_transpose"] = {"bit_exact": bool(torch.equal(g.reshape(x.transpose(-2,-1).shape), x.transpose(-2, -1)))}
except Exception as e:
    R["rowmajor_transpose"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
print("rowmajor transpose", R["rowmajor_transpose"], flush=True)
# exact intra-tile movement candidates that avoid the FPU
for nm, fn in (("concat_slice_16", lambda t: ttnn.concat([ttnn.slice(t, [0,0,0,16], [1,1,256,32]), ttnn.slice(t, [0,0,0,0], [1,1,256,16])], dim=-1)),
               ("roll_16", lambda t: ttnn.roll(t, 16, -1) if hasattr(ttnn, "roll") else None)):
    try:
        g = fn(tx)
        R[nm] = {"bit_exact": bool(torch.equal(ttnn.to_torch(g), torch.cat([x[..., 16:], x[..., :16]], -1)))} if g is not None else {"absent": True}
    except Exception as e:
        R[nm] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    print(nm, R[nm], flush=True)
json.dump(R, open("/home/ttuser/.coworker/wt/ttnn-fft-kernel-spike/fftprobe/probe_p4.json", "w"), indent=1)
print("WROTE probe_p4.json")
ttnn.close_device(dev)
