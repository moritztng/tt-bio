"""Split the 151.2 ms bridge call into projection vs compare, so the value of putting only the
compare on device is a measured number rather than an assumption."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/home/ttuser/.coworker/wt/relion-acc-backend")
import tt_bio.cryoem.relion as R

torch.set_grad_enabled(False)
mdlX, mdlY, mdlZ = 100, 199, 199
mdlInitY, mdlInitZ = -99, -99
maxR, pf = 98, 2.0
maxR2p = int(maxR * maxR * pf * pf)
imgX, imgY = 99, 196
P = imgX * imgY
No, Nt = 186, 9
CH = 32

rng = np.random.default_rng(0)
mdl = torch.from_numpy((rng.standard_normal(mdlX * mdlY * mdlZ * 2).astype(np.float32) * 0.01)
                       .reshape(-1, 2))
eul = torch.empty(No, 9)
for i in range(No):
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    eul[i] = torch.from_numpy(q.reshape(-1).astype(np.float32))
tx = torch.from_numpy((rng.standard_normal(Nt) * 0.05).astype(np.float32))
ty = torch.from_numpy((rng.standard_normal(Nt) * 0.05).astype(np.float32))
img_r = torch.from_numpy(rng.standard_normal(P).astype(np.float32))
img_i = torch.from_numpy(rng.standard_normal(P).astype(np.float32))
w = torch.from_numpy(np.abs(rng.standard_normal(P)).astype(np.float32)) * 0.5

pix = torch.arange(P, dtype=torch.int64)
x = (pix % imgX).float()
yi = pix // imgX
y = torch.where(yi > maxR, yi - imgY, yi).float()
ph = x.unsqueeze(0) * tx.unsqueeze(1) + y.unsqueeze(0) * ty.unsqueeze(1)
s, c = torch.sin(ph), torch.cos(ph)
sh_r = c * img_r - s * img_i
sh_i = c * img_i + s * img_r

REPS = 5


def time_it(fn, reps=REPS):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


def proj_only():
    for o0 in range(0, No, CH):
        R._project(torch, mdl, x, y, eul[o0:o0 + CH], mdlX, mdlY, mdlZ,
                   mdlInitY, mdlInitZ, maxR2p, pf)


refs = [R._project(torch, mdl, x, y, eul[o0:o0 + CH], mdlX, mdlY, mdlZ,
                   mdlInitY, mdlInitZ, maxR2p, pf) for o0 in range(0, No, CH)]


def compare_direct():
    for rr, ri in refs:
        dr = rr.unsqueeze(1) - sh_r.unsqueeze(0)
        di = ri.unsqueeze(1) - sh_i.unsqueeze(0)
        ((dr * dr + di * di) * w).sum(-1)


def compare_gemm():
    # A[o] + B - 2 C[o,t], the two-real-matmul form the estep pass measured at the matmul roof
    B = (w * (img_r * img_r + img_i * img_i)).sum()
    for rr, ri in refs:
        A = (w * (rr * rr + ri * ri)).sum(-1)
        C = (rr * w) @ sh_r.T + (ri * w) @ sh_i.T
        A.unsqueeze(1) + B - 2.0 * C


tp, cd, cg = time_it(proj_only), time_it(compare_direct), time_it(compare_gemm)
print("torch threads          %d" % torch.get_num_threads())
print("projection (trilinear) %7.1f ms" % tp)
print("compare, direct form   %7.1f ms" % cd)
print("compare, GEMM form     %7.1f ms" % cg)
print("total direct           %7.1f ms   projection is %4.1f%%" % (tp + cd, 100 * tp / (tp + cd)))
print("total GEMM             %7.1f ms   projection is %4.1f%%" % (tp + cg, 100 * tp / (tp + cg)))

# does the GEMM form agree with the direct form? it is a different summation, so check it.
mx = 0.0
Bt = (w * (img_r * img_r + img_i * img_i)).sum()
for rr, ri in refs:
    dr = rr.unsqueeze(1) - sh_r.unsqueeze(0)
    di = ri.unsqueeze(1) - sh_i.unsqueeze(0)
    d_direct = ((dr * dr + di * di) * w).sum(-1)
    A = (w * (rr * rr + ri * ri)).sum(-1)
    C = (rr * w) @ sh_r.T + (ri * w) @ sh_i.T
    d_gemm = A.unsqueeze(1) + Bt - 2.0 * C
    rel = (d_gemm - d_direct).abs() / d_direct.abs().clamp_min(1e-30)
    mx = max(mx, float(rel.max()))
print("GEMM vs direct, max relative difference on diff2: %.3e" % mx)
print("gathers per call: %d  (No x P x 8 corners)" % (No * P * 8))
