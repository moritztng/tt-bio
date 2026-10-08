"""where are fid32.py's HiFi4 max-error outliers? WH .107 chip 10 2026-10-08: 2-5 elements per 11M outputs at
K>=768 off by exactly -2^k (-2, -0.5, -4), both core grids, M=1 and 5, 730 and 736 rows; HiFi3: none."""
import sys, torch
import ttnn, tt_bio.tenstorrent as T
dev = T.get_device()
g = torch.Generator().manual_seed(0)
up = lambda t, dt=ttnn.float32: ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
for (M, R, K, N) in [(5, 730, 768, 3072), (5, 730, 1536, 768), (1, 730, 768, 3072), (5, 736, 768, 3072)]:
    x_h = torch.randn(M, R, K, generator=g); w_h = torch.randn(K, N, generator=g) * K ** -0.5
    y64 = x_h.double() @ w_h.double(); x, w = up(x_h), up(w_h)
    for fid in ("HiFi4", "HiFi3"):
        for cg in ("main", "none"):
            ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                         fp32_dest_acc_en=True, packer_l1_acc=True)
            kw = dict(core_grid=T.CORE_GRID_MAIN) if cg == "main" else {}
            y = torch.Tensor(ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=ckc, **kw))).double()
            d = (y - y64).abs(); bad = (d > 0.05).nonzero()
            print(f"MRKN={M,R,K,N} {fid} grid={cg} max={float(d.max()):.4g} mean={float(d.mean()):.4g} n_bad={len(bad)}",
                  "first:", bad[:6].tolist(), "vals y/y64:", [(round(float(y[tuple(b)]), 4), round(float(y64[tuple(b)]), 4)) for b in bad[:3]],
                  "bad rows:", sorted(set(bad[:, 1].tolist()))[:10], "bad cols//32:", sorted(set((bad[:, 2] // 32).tolist()))[:10], flush=True)
