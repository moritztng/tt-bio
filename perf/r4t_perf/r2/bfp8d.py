import torch, ttnn
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
from tt_bio import tenstorrent as T
dev = T.get_device()
g = torch.Generator().manual_seed(0)
tot = 0; nd = 0; better = 0; worse = 0; shown = 0
for i in range(20):
    x = torch.randn(1024, 1280, generator=g) * 0.02
    h = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)).float()
    d = ttnn.to_torch(ttnn.typecast(ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev), ttnn.bfloat8_b)).float()
    m = h != d; tot += x.numel(); nd += int(m.sum())
    eh = (h - x).abs()[m]; ed = (d - x).abs()[m]
    better += int((ed < eh).sum()); worse += int((ed > eh).sum())
    for idx in m.nonzero()[:max(0, 6 - shown)]:
        r, c = idx.tolist(); c0 = (c // 16) * 16
        blk = x[r, c0:c0 + 16]
        print("x %.9e host %.9e dev %.9e blockmax %.6e  ratio x/ulp %.4f" % (x[r, c], h[r, c], d[r, c], blk.abs().max(),
              (x[r, c] / ((h[r, c] - d[r, c]).abs())).item()), flush=True)
        shown += 1
print("diff %d of %d (%.2e); dev closer %d, host closer %d, tie %d" % (nd, tot, nd / tot, better, worse, nd - better - worse))
