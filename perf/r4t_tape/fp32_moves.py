"""fp32 channel moves and batched matmuls vs torch, exact where they should be: the fp32 incoming trimul reads 0.084 off float64."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import reblock_permute as R
    dev = tt.get_device()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 64, 64, 256, generator=g)
    for dt, name in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
        xd = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        xr = ttnn.to_torch(xd).float()
        for p in ((0, 3, 1, 2), (0, 3, 2, 1), (0, 2, 1, 3), (0, 2, 3, 1), (0, 1, 3, 2)):
            got = ttnn.to_torch(ttnn.permute(xd, p)).float()
            print(name, "permute", p, "max|d|", float((got - xr.permute(p)).abs().max()), flush=True)
        mc = xd.memory_config()
        print(name, "reblock eligible", R.eligible(xd, mc), "back", R.eligible_back(xd, mc), flush=True)
        if R.eligible(xd, mc):
            print(name, "reblock_permute", float((ttnn.to_torch(R.reblock_permute(xd, mc)).float() - xr.permute(0, 3, 1, 2)).abs().max()), flush=True)
        a = torch.randn(1, 256, 64, 64, generator=g)
        b = torch.randn(1, 256, 64, 64, generator=g)
        ad = ttnn.from_torch(a, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        bd = ttnn.from_torch(b, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        ar, br = ttnn.to_torch(ad).double(), ttnn.to_torch(bd).double()
        for ta, tb in ((False, True), (True, False)):
            got = ttnn.to_torch(ttnn.matmul(ad, bd, transpose_a=ta, transpose_b=tb)).double()
            want = (ar.transpose(-1, -2) if ta else ar) @ (br.transpose(-1, -2) if tb else br)
            print(name, "matmul ta", ta, "tb", tb, "rel", float((got - want).norm() / want.norm()), flush=True)


if __name__ == "__main__":
    main()
