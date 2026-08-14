#!/usr/bin/env python3
"""Can a split-bf16 (double-single) matmul break the ~1.1e-3 fp32-matmul wall?
The wall is what blocked tt-metal #21412 for 15 months ("the basic FFT is trivial but the
precision is terrible"). p2 tested this wrong: bf16 OUTPUT dtype rounds the correction terms
away. Here every product is accumulated with dtype=float32 and fp32_dest_acc_en."""
import json
import numpy as np, torch, ttnn

R = {}
dev = ttnn.open_device(device_id=0)
def cfg(fid, acc=True):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid),
                                            math_approx_mode=False, fp32_dest_acc_en=acc,
                                            packer_l1_acc=False)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
def split_bf16(v):
    hi = v.to(torch.bfloat16).float()
    lo = (v - hi).to(torch.bfloat16).float()
    return hi, lo

torch.manual_seed(0)
for K in (32, 128, 1024):
    a = torch.randn(1, 1, 32, K, dtype=torch.float32)
    b = torch.randn(1, 1, K, 32, dtype=torch.float32)
    ref = (a.double() @ b.double()).numpy()
    key = f"K{K}"; R[key] = {}

    ta = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    for fid in ("HiFi4", "HiFi3"):
        R[key][f"fp32_in_{fid}"] = rel(ttnn.to_torch(ttnn.matmul(
            ta, tb, compute_kernel_config=cfg(fid), dtype=ttnn.float32)).double().numpy(), ref)

    ah, al = split_bf16(a); bh, bl = split_bf16(b)
    def bf(v): return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    dah, dal, dbh, dbl = bf(ah), bf(al), bf(bh), bf(bl)
    def mm(x, y, fid="HiFi4"):
        return ttnn.matmul(x, y, compute_kernel_config=cfg(fid), dtype=ttnn.float32)
    hh, hl, lh, ll = mm(dah, dbh), mm(dah, dbl), mm(dal, dbh), mm(dal, dbl)
    g1 = ttnn.to_torch(hh).double().numpy()
    g3 = ttnn.to_torch(ttnn.add(ttnn.add(hh, hl), lh)).double().numpy()
    g4 = ttnn.to_torch(ttnn.add(ttnn.add(ttnn.add(hh, hl), lh), ll)).double().numpy()
    R[key]["split1_hh_only"] = rel(g1, ref)
    R[key]["split3_products"] = rel(g3, ref)
    R[key]["split4_products"] = rel(g4, ref)
    # what a host float32 matmul gives, as the achievable-precision reference
    R[key]["host_fp32_matmul"] = rel((a @ b).double().numpy(), ref)
    print(K, json.dumps(R[key], indent=1), flush=True)

# ---- end to end: radix-32 four-step at N=1024 with the 3-product split matmul ----
B, N, S = 64, 1024, 32
rng = np.random.default_rng(0)
x = rng.standard_normal((B, N)) + 1j * rng.standard_normal((B, N))
ref = np.fft.fft(x.astype(np.complex128), axis=-1)
n1 = np.arange(S)[:, None]; k1 = np.arange(S)[None, :]
F = np.exp(-2j * np.pi * n1 * k1 / S)
Tw = np.exp(-2j * np.pi * (np.arange(S)[:, None] * np.arange(S)[None, :]) / N)

def T(a, dt=ttnn.float32):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)).float(), dtype=dt,
                           layout=ttnn.TILE_LAYOUT, device=dev)
def Tsplit(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).float()
    hi, lo = split_bf16(t)
    return (ttnn.from_torch(hi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            ttnn.from_torch(lo, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev))

def split_mm(A, Bm):
    """3-product split-bf16 real matmul, fp32 accumulate. A,Bm are (hi,lo) pairs."""
    ah, al = A; bh, bl = Bm
    p = ttnn.matmul(ah, bh, compute_kernel_config=cfg("HiFi4"), dtype=ttnn.float32)
    p = ttnn.add(p, ttnn.matmul(ah, bl, compute_kernel_config=cfg("HiFi4"), dtype=ttnn.float32))
    p = ttnn.add(p, ttnn.matmul(al, bh, compute_kernel_config=cfg("HiFi4"), dtype=ttnn.float32))
    return p

def cmm_split(Ar, Ai, Br, Bi):
    return (ttnn.sub(split_mm(Ar, Br), split_mm(Ai, Bi)),
            ttnn.add(split_mm(Ar, Bi), split_mm(Ai, Br)))

X = x.reshape(B, 1, S, S)
Fk = np.broadcast_to(F.T.reshape(1, 1, S, S), (B, 1, S, S))
FT = np.broadcast_to(F.reshape(1, 1, S, S), (B, 1, S, S))
Tb = np.broadcast_to(Tw.reshape(1, 1, S, S), (B, 1, S, S))
Xr, Xi = Tsplit(X.real), Tsplit(X.imag)
Fr, Fi = Tsplit(Fk.real), Tsplit(Fk.imag)
Gr, Gi = Tsplit(FT.real), Tsplit(FT.imag)
wr, wi = T(Tb.real), T(Tb.imag)

ar, ai = cmm_split(Fr, Fi, Xr, Xi)                      # stage 1
br = ttnn.sub(ttnn.mul(ar, wr), ttnn.mul(ai, wi))       # fp32 eltwise: measured bit-exact
bi = ttnn.add(ttnn.mul(ar, wi), ttnn.mul(ai, wr))
Brs, Bis = Tsplit(ttnn.to_torch(br).numpy()), Tsplit(ttnn.to_torch(bi).numpy())
cr, ci = cmm_split(Brs, Bis, Gr, Gi)                    # stage 2
C = ttnn.to_torch(cr).double().numpy() + 1j * ttnn.to_torch(ci).double().numpy()
got = np.transpose(C.reshape(B, S, S), (0, 2, 1)).reshape(B, N)
R["radix32_N1024_split3"] = {"rel_l2": rel(got, ref),
                             "max_abs_over_max_ref": float(np.abs(got - ref).max() / np.abs(ref).max())}
print("radix32 N=1024 split-3", R["radix32_N1024_split3"], flush=True)

json.dump(R, open("/home/ttuser/.coworker/wt/ttnn-fft-kernel-spike/fftprobe/probe_p3.json", "w"), indent=1)
print("WROTE probe_p3.json")
ttnn.close_device(dev)
