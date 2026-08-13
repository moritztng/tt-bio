#!/usr/bin/env python3
"""Where does fp32 precision actually go on Blackhole? Isolate storage, eltwise, matmul, transpose.
Then test the split-bf16 (bf16x2, 3-product) matmul as the escape from the ~1e-3 wall."""
import json, time
import numpy as np, torch, ttnn

R = {}
dev = ttnn.open_device(device_id=0)
def sync(): ttnn.synchronize_device(dev)
def cfg(fid, acc):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                            fp32_dest_acc_en=acc, packer_l1_acc=False)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))

torch.manual_seed(0)
# ---- storage round trip ----
x = torch.randn(512, 512, dtype=torch.float32)
t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
R["storage_roundtrip_bit_exact"] = bool(torch.equal(ttnn.to_torch(t), x))
R["storage_roundtrip_rel"] = rel(ttnn.to_torch(t).double().numpy(), x.double().numpy())
print("storage", R["storage_roundtrip_bit_exact"], R["storage_roundtrip_rel"], flush=True)

# ---- eltwise mul/add/sub in fp32 ----
y = torch.randn(512, 512, dtype=torch.float32)
u = ttnn.from_torch(y, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
for nm, f, ref in [("mul", ttnn.mul, x.double() * y.double()),
                   ("add", ttnn.add, x.double() + y.double()),
                   ("sub", ttnn.sub, x.double() - y.double())]:
    g = ttnn.to_torch(f(t, u)).double().numpy()
    R["eltwise_" + nm] = {"rel": rel(g, ref.numpy()),
                          "bit_exact_vs_fp32": bool(torch.equal(ttnn.to_torch(f(t, u)), f.__call__ and (x * y if nm == "mul" else (x + y if nm == "add" else x - y))))}
    print("eltwise", nm, R["eltwise_" + nm], flush=True)

# ---- matmul, K=32, every fidelity x fp32_dest_acc ----
K = 32
a32 = torch.randn(1, 1, 32, K, dtype=torch.float32)
b32 = torch.randn(1, 1, K, 32, dtype=torch.float32)
refm = (a32.double() @ b32.double()).numpy()
ta = ttnn.from_torch(a32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
tb = ttnn.from_torch(b32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
R["matmul_k32"] = {}
for fid in ("LoFi", "HiFi2", "HiFi3", "HiFi4"):
    for acc in (False, True):
        try:
            g = ttnn.to_torch(ttnn.matmul(ta, tb, compute_kernel_config=cfg(getattr(ttnn.MathFidelity, fid), acc))).double().numpy()
            R["matmul_k32"][f"{fid}_acc{int(acc)}"] = rel(g, refm)
        except Exception as e:
            R["matmul_k32"][f"{fid}_acc{int(acc)}"] = f"ERR {type(e).__name__}"
print("matmul K=32", json.dumps(R["matmul_k32"], indent=1), flush=True)

# ---- the escape: split-bf16 matmul (a = a_hi + a_lo, 3 products) ----
def split_bf16(v):
    hi = v.to(torch.bfloat16).float()
    lo = (v - hi).to(torch.bfloat16).float()
    return hi, lo
ah, al = split_bf16(a32); bh, bl = split_bf16(b32)
def dv(v): return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
lofi = cfg(ttnn.MathFidelity.LoFi, True)
p_hh = ttnn.matmul(dv(ah), dv(bh), compute_kernel_config=lofi)
p_hl = ttnn.matmul(dv(ah), dv(bl), compute_kernel_config=lofi)
p_lh = ttnn.matmul(dv(al), dv(bh), compute_kernel_config=lofi)
p_ll = ttnn.matmul(dv(al), dv(bl), compute_kernel_config=lofi)
s3 = ttnn.to_torch(ttnn.add(ttnn.add(p_hh, p_hl), p_lh)).double().numpy()
s4 = ttnn.to_torch(ttnn.add(ttnn.add(ttnn.add(p_hh, p_hl), p_lh), p_ll)).double().numpy()
R["split_bf16_3prod_k32"] = rel(s3, refm)
R["split_bf16_4prod_k32"] = rel(s4, refm)
R["bf16_1prod_k32"] = rel(ttnn.to_torch(p_hh).double().numpy(), refm)
print("split-bf16 K=32: 1prod", R["bf16_1prod_k32"], "3prod", R["split_bf16_3prod_k32"], "4prod", R["split_bf16_4prod_k32"], flush=True)

# ---- does the 3-product split fix the twiddle multiply too? ----
xh, xl = split_bf16(x); yh, yl = split_bf16(y)
m3 = ttnn.to_torch(ttnn.add(ttnn.add(ttnn.mul(dv(xh), dv(yh)), ttnn.mul(dv(xh), dv(yl))), ttnn.mul(dv(xl), dv(yh)))).double().numpy()
R["split_bf16_3prod_eltwise_mul"] = rel(m3, (x.double() * y.double()).numpy())
print("split eltwise mul", R["split_bf16_3prod_eltwise_mul"], flush=True)

json.dump(R, open("/home/ttuser/.coworker/wt/ttnn-fft-kernel-spike/fftprobe/probe_p2.json", "w"), indent=1)
print("WROTE probe_p2.json")
ttnn.close_device(dev)
