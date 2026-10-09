"""HiFi4 fp32 linear power-of-two outliers split by fp32_dest_acc / packer_l1_acc. WH .107 chip 10 2026-10-08:
HiFi4+fp32 dest 4 (packer L1 on) / 2 (off) outliers, deterministic; HiFi4+bf16 dest and HiFi3+fp32 dest none; HiFi3 same speed."""
import time, torch
import ttnn, tt_bio.tenstorrent as T
dev = T.get_device()
g = torch.Generator().manual_seed(0)
up = lambda t: ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
M, R, K, N = 5, 730, 768, 3072
x_h = torch.randn(M, R, K, generator=g); w_h = torch.randn(K, N, generator=g) * K ** -0.5
y64 = x_h.double() @ w_h.double(); x, w = up(x_h), up(w_h)
for fid, acc, pk in [("HiFi4", True, True), ("HiFi4", True, False), ("HiFi4", False, False), ("HiFi3", True, False), ("HiFi4", True, True)]:
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                 fp32_dest_acc_en=acc, packer_l1_acc=pk)
    f = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
    ys = [torch.Tensor(ttnn.to_torch(f())).double() for _ in range(3)]
    ttnn.synchronize_device(dev); t0 = time.perf_counter()
    for _ in range(10): ttnn.deallocate(f())
    ttnn.synchronize_device(dev); us = (time.perf_counter() - t0) / 10 * 1e6
    d = (ys[0] - y64).abs(); bad = (d > 0.05).nonzero()
    print(f"{fid} dest_acc={acc} packer_l1={pk} us={us:.0f} max={float(d.max()):.4g} mean={float(d.mean()):.4g} n_bad={len(bad)}"
          f" deterministic={all(torch.equal(ys[0], y) for y in ys)} offsets={[round(float(ys[0][tuple(b)] - y64[tuple(b)]), 3) for b in bad[:6]]}", flush=True)
