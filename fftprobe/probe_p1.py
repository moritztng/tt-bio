#!/usr/bin/env python3
"""Phase-1 probes for the tile-native radix-32 FFT design.

1. DRAM bandwidth roof on THIS card (eltwise add, 3-tensor traffic model).
2. fp32 transpose fidelity -- the obstacle jasondavies named on tt-metal #21412 (2025-12-17):
   "the precision is seriously affected by the lack of full precision fp32 transpose".
3. Accuracy of the radix-32 four-step (F32 @ (X o T) @ F32) at N=1024, fp32 and bf16,
   against the dense N-point DFT-by-matmul that the feasibility pass measured.
"""
import json, math, time
import numpy as np, torch, ttnn

R = {}
dev = ttnn.open_device(device_id=0)
def sync(): ttnn.synchronize_device(dev)

FP32_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=False)
LOFI_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=False)

# ---------------- 1. bandwidth roof ----------------
def bw_roof(dtype, nbytes_el, n=8192):
    a = ttnn.from_torch(torch.randn(n, n), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(n, n), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    o = ttnn.add(a, b); sync(); ttnn.deallocate(o)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); o = ttnn.add(a, b); sync(); dt = time.perf_counter() - t0
        ttnn.deallocate(o); best = min(best, dt)
    gbs = 3 * n * n * nbytes_el / best / 1e9
    ttnn.deallocate(a); ttnn.deallocate(b)
    return {"ms": best * 1e3, "GB_s": gbs, "n": n}

R["bw_fp32"] = bw_roof(ttnn.float32, 4)
R["bw_bf16"] = bw_roof(ttnn.bfloat16, 2)
print("BW fp32", R["bw_fp32"]); print("BW bf16", R["bw_bf16"], flush=True)

# ---------------- 2. fp32 transpose fidelity ----------------
xt = torch.randn(4, 1, 512, 512, dtype=torch.float32)
tt = ttnn.from_torch(xt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
for name, fn in [("transpose_-2-1", lambda t: ttnn.transpose(t, -2, -1)),
                 ("permute_0132",   lambda t: ttnn.permute(t, (0, 1, 3, 2)))]:
    try:
        y = ttnn.to_torch(fn(tt)); ref = xt.transpose(-2, -1)
        R["fp32_" + name] = {"bit_exact": bool(torch.equal(y, ref)),
                             "max_abs_err": float((y - ref).abs().max()),
                             "rel_l2": float((y - ref).norm() / ref.norm())}
    except Exception as e:
        R["fp32_" + name] = {"error": f"{type(e).__name__}: {e}"}
    print("transpose", name, R["fp32_" + name], flush=True)
ttnn.deallocate(tt)

# ---------------- 3. accuracy of the radix-32 four-step at N=1024 ----------------
B, N, S = 64, 1024, 32
rng = np.random.default_rng(0)
x = rng.standard_normal((B, N)) + 1j * rng.standard_normal((B, N))
ref = np.fft.fft(x.astype(np.complex128), axis=-1)          # float64 reference

def to_dev(a, dtype):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)).float(),
                           dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)

def cmatmul(ar, ai, br, bi, cfg):
    rr = ttnn.matmul(ar, br, compute_kernel_config=cfg)
    ii = ttnn.matmul(ai, bi, compute_kernel_config=cfg)
    ri = ttnn.matmul(ar, bi, compute_kernel_config=cfg)
    ir = ttnn.matmul(ai, br, compute_kernel_config=cfg)
    out_r = ttnn.sub(rr, ii); out_i = ttnn.add(ri, ir)
    for t in (rr, ii, ri, ir): ttnn.deallocate(t)
    return out_r, out_i

def cmul(ar, ai, br, bi):
    rr = ttnn.mul(ar, br); ii = ttnn.mul(ai, bi)
    ri = ttnn.mul(ar, bi); ir = ttnn.mul(ai, br)
    out_r = ttnn.sub(rr, ii); out_i = ttnn.add(ri, ir)
    for t in (rr, ii, ri, ir): ttnn.deallocate(t)
    return out_r, out_i

# F_32 and the four-step twiddle, both precomputed on host in float64 then cast
n1 = np.arange(S)[:, None]; k1 = np.arange(S)[None, :]
F = np.exp(-2j * np.pi * n1 * k1 / S)                        # F[n,k]; F32[k1,n1] = F.T
Tw = np.exp(-2j * np.pi * (np.arange(S)[:, None] * np.arange(S)[None, :]) / N)   # T[k1,n2]

def radix32_fourstep(dtype, cfg, label):
    X = x.reshape(B, 1, S, S)                                # X[b, n1, n2]
    Fk = np.broadcast_to(F.T.reshape(1, 1, S, S), (B, 1, S, S))   # F32[k1,n1]
    FT = np.broadcast_to(F.reshape(1, 1, S, S), (B, 1, S, S))     # F[n2,k2] = F32[k2,n2]^T
    Tb = np.broadcast_to(Tw.reshape(1, 1, S, S), (B, 1, S, S))
    xr, xi = to_dev(X.real, dtype), to_dev(X.imag, dtype)
    fr, fi = to_dev(Fk.real, dtype), to_dev(Fk.imag, dtype)
    tr, ti = to_dev(FT.real, dtype), to_dev(FT.imag, dtype)
    wr, wi = to_dev(Tb.real, dtype), to_dev(Tb.imag, dtype)
    sync()
    def chain():
        ar, ai = cmatmul(fr, fi, xr, xi, cfg)                # stage 1: length-32 over n1
        br, bi = cmul(ar, ai, wr, wi)                        # four-step twiddle
        ttnn.deallocate(ar); ttnn.deallocate(ai)
        cr, ci = cmatmul(br, bi, tr, ti, cfg)                # stage 2: length-32 over n2
        ttnn.deallocate(br); ttnn.deallocate(bi)
        return cr, ci
    cr, ci = chain(); sync()
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); a, b_ = chain(); sync(); dt = time.perf_counter() - t0
        ttnn.deallocate(a); ttnn.deallocate(b_); best = min(best, dt)
    C = ttnn.to_torch(cr).double().numpy() + 1j * ttnn.to_torch(ci).double().numpy()
    C = C.reshape(B, S, S)                                   # C[b, k1, k2]
    got = np.transpose(C, (0, 2, 1)).reshape(B, N)            # k = k2*32 + k1
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    out = {"rel_l2": float(err), "max_abs_over_max_ref": float(np.abs(got - ref).max() / np.abs(ref).max()),
           "ms": best * 1e3, "transforms_s": B / best}
    print("radix32", label, out, flush=True)
    for t in (xr, xi, fr, fi, tr, ti, wr, wi, cr, ci): ttnn.deallocate(t)
    return out

R["radix32_fp32"] = radix32_fourstep(ttnn.float32, FP32_CFG, "fp32/HiFi4")
R["radix32_bf16"] = radix32_fourstep(ttnn.bfloat16, LOFI_CFG, "bf16/LoFi")

# dense N=1024 DFT by matmul, same batch -- the feasibility path, for the error+cost contrast
def dense_dft(dtype, cfg, label):
    n = np.arange(N)[:, None]; k = np.arange(N)[None, :]
    FN = np.exp(-2j * np.pi * n * k / N)                     # FN[n,k]
    xr, xi = to_dev(x.real.reshape(1, 1, B, N), dtype), to_dev(x.imag.reshape(1, 1, B, N), dtype)
    fr, fi = to_dev(FN.real.reshape(1, 1, N, N), dtype), to_dev(FN.imag.reshape(1, 1, N, N), dtype)
    sync()
    yr, yi = cmatmul(xr, xi, fr, fi, cfg); sync()
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); a, b_ = cmatmul(xr, xi, fr, fi, cfg); sync(); dt = time.perf_counter() - t0
        ttnn.deallocate(a); ttnn.deallocate(b_); best = min(best, dt)
    got = (ttnn.to_torch(yr).double().numpy() + 1j * ttnn.to_torch(yi).double().numpy()).reshape(B, N)
    out = {"rel_l2": float(np.linalg.norm(got - ref) / np.linalg.norm(ref)),
           "ms": best * 1e3, "transforms_s": B / best}
    print("dense", label, out, flush=True)
    for t in (xr, xi, fr, fi, yr, yi): ttnn.deallocate(t)
    return out

R["dense1024_fp32"] = dense_dft(ttnn.float32, FP32_CFG, "fp32/HiFi4")
R["dense1024_bf16"] = dense_dft(ttnn.bfloat16, LOFI_CFG, "bf16/LoFi")

json.dump(R, open("/home/ttuser/.coworker/wt/ttnn-fft-kernel-spike/fftprobe/probe_p1.json", "w"), indent=1)
print("WROTE probe_p1.json")
ttnn.close_device(dev)
