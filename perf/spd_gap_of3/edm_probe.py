"""Probe one edm_device step's device ops stage by stage against torch (float64)."""
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import math, torch, ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402

dev = T.get_device()
try:
    g = torch.Generator().manual_seed(0)
    n, NP, sd, t, ct, ss = 4100, 4128, 16.0, 80.0, 70.0, 1.5
    m = torch.zeros(NP); m[:n] = (torch.rand(n, generator=g) > 0.05).float()
    x = torch.zeros(NP, 3); x[:n] = torch.randn(n, 3, generator=g) * 30
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g)); tr = torch.randn(3, generator=g)
    nz = torch.zeros(NP, 3); nz[:n] = torch.randn(n, 3, generator=g) * t
    ru_h = torch.zeros(NP, 3); ru_h[:n] = torch.randn(n, 3, generator=g)
    up = lambda a: ttnn.from_torch(a.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    rd = lambda a: ttnn.to_torch(a).double()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=False)
    inv_s = 1 / math.sqrt(t * t + sd * sd); cs = sd * sd / (sd * sd + t * t)
    co = sd * t / math.sqrt(sd * sd + t * t); inv_t = 1 / t; h = ss * (ct - t)
    S = [up(torch.tensor(v).reshape(1, 1, 1)) for v in (inv_s, cs, co, inv_t, h)]
    md, mrow = up(m.reshape(1, NP, 1)), up((m / m.sum()).reshape(1, 1, NP))
    X, RT, TR, NZ, RU = up(x[None]), up(q.t()[None]), up(tr.reshape(1, 1, 3)), up(nz[None]), up(ru_h[None])
    D = x.double(); M = m.double()[:, None]
    def chk(name, d, ref):
        a = rd(d).reshape(ref.shape)
        print(f"{name:10s} max_abs={float((a - ref).abs().max()):.3e} ref_max={float(ref.abs().max()):.3e} shape={tuple(d.shape)}", flush=True)
    mean_r = (D * M).sum(0, keepdim=True) / M.sum()
    mean = ttnn.matmul(mrow, X, compute_kernel_config=ckc); chk("mean", mean, mean_r)
    xc = ttnn.subtract(X, mean); xc_r = D - mean_r; chk("centred", xc, xc_r)
    xr = ttnn.matmul(xc, RT, compute_kernel_config=ckc); xr_r = xc_r @ q.t().double(); chk("rotated", xr, xr_r)
    xa = ttnn.add(xr, TR); xa_r = xr_r + tr.double(); chk("translated", xa, xa_r)
    xam = ttnn.multiply(xa, md); xam_r = xa_r * M; chk("masked", xam, xam_r)
    xn = ttnn.add(xam, NZ); xn_r = xam_r + nz.double(); chk("noisy", xn, xn_r)
    xnm = ttnn.multiply(xn, md); xnm_r = xn_r * M; chk("noisy_m", xnm, xnm_r)
    rl = ttnn.multiply(xnm, S[0]); chk("scaled", rl, xnm_r * inv_s)
    a = ttnn.multiply(xnm, S[1]); chk("c_skip", a, xnm_r * cs)
    b = ttnn.multiply(RU, S[2]); chk("c_out", b, ru_h.double() * co)
    xd = ttnn.multiply(ttnn.add(a, b), md); xd_r = (xnm_r * cs + ru_h.double() * co) * M; chk("denoised", xd, xd_r)
    d = ttnn.multiply(ttnn.subtract(xn, xd), S[3]); d_r = (xn_r - xd_r) * inv_t; chk("delta", d, d_r)
    xnew = ttnn.add(xn, ttnn.multiply(d, S[4])); chk("update", xnew, xn_r + d_r * h)
    buf = up(torch.zeros(1, NP, 3)); ttnn.copy(xnew, buf); chk("copy", buf, xn_r + d_r * h)
    # pad check (non-aligned)
    r1 = up(ru_h[:n][None]); rp = ttnn.pad(r1, padding=[(0, 0), (0, NP - n), (0, 0)], value=0.0)
    chk("pad", rp, ru_h.double()); 
    # broadcast via python-float scalar for comparison
    chk("scal_float", ttnn.multiply(xnm, cs), xnm_r * cs)
finally:
    T.cleanup()
