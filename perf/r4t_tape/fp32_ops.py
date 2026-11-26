"""fp32 op floors vs float64 on the card: which ops a 'genuinely fp32' PairUpdate can rely on (D49(2))."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio.rfd4 import trunk as TR
    dev = tt.get_device()
    ckc = TR.compute_kernel_config()
    g = torch.Generator().manual_seed(0)
    f32 = ttnn.float32

    def up(t):
        return ttnn.from_torch(t.float().contiguous(), dtype=f32, layout=ttnn.TILE_LAYOUT, device=dev)

    def down(t):
        return ttnn.to_torch(t).double()

    x = torch.randn(1, 64, 64, 128, generator=g, dtype=torch.float64)
    xr = x.float().double()  # what the card holds
    xd = up(x)
    print("roundtrip", rel(down(xd), xr), flush=True)
    for p in ((0, 3, 1, 2), (0, 2, 1, 3), (0, 1, 3, 2)):
        print("permute", p, rel(down(ttnn.permute(xd, p)), xr.permute(p)), flush=True)
    print("transpose(-2,-1)", rel(down(ttnn.transpose(xd, -2, -1)), xr.transpose(-2, -1)), flush=True)
    w = torch.randn(128, 128, generator=g, dtype=torch.float64) / 128 ** 0.5
    wr = w.float().double()
    wd = up(w)
    print("matmul ckc", rel(down(ttnn.matmul(xd, wd, compute_kernel_config=ckc)), xr @ wr), flush=True)
    print("matmul default", rel(down(ttnn.matmul(xd, wd)), xr @ wr), flush=True)
    print("linear ckc", rel(down(ttnn.linear(xd, wd, compute_kernel_config=ckc)), xr @ wr), flush=True)
    a = torch.randn(1, 128, 64, 64, generator=g, dtype=torch.float64)
    ar = a.float().double()
    ad = up(a)
    print("bmm ckc", rel(down(ttnn.matmul(ad, ad, compute_kernel_config=ckc)), ar @ ar), flush=True)
    print("bmm tb ckc", rel(down(ttnn.matmul(ad, ad, transpose_b=True, compute_kernel_config=ckc)), ar @ ar.transpose(-1, -2)), flush=True)
    lw, lb = torch.randn(128, generator=g, dtype=torch.float64), torch.randn(128, generator=g, dtype=torch.float64)
    ref = torch.nn.functional.layer_norm(xr, (128,), lw.float().double(), lb.float().double(), 1e-5)
    print("layer_norm ckc", rel(down(ttnn.layer_norm(xd, weight=up(lw.reshape(1, -1)), bias=up(lb.reshape(1, -1)), epsilon=1e-5, compute_kernel_config=ckc)), ref), flush=True)
    y = torch.randn(1, 64, 64, 128, generator=g, dtype=torch.float64)
    yr, yd = y.float().double(), up(y)
    print("multiply", rel(down(ttnn.multiply(xd, yd)), xr * yr), flush=True)
    print("add", rel(down(ttnn.add(xd, yd)), xr + yr), flush=True)
    print("relu", rel(down(ttnn.relu(xd)), xr.clamp(min=0)), flush=True)
    print("sigmoid", rel(down(ttnn.sigmoid(xd)), torch.sigmoid(xr)), flush=True)
    print("sum(-1)", rel(down(ttnn.sum(xd, dim=-1, keepdim=True)), xr.sum(-1, keepdim=True)), flush=True)


if __name__ == "__main__":
    main()
