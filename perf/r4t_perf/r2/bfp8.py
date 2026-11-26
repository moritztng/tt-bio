import os, time, numpy as np
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
dev = T.get_device()
g = torch.Generator().manual_seed(0)
for shape in [(384, 1536), (768, 768), (256, 6144), (32, 64)]:
    x = torch.randn(*shape, generator=g) * 0.02
    x[0, :5] = torch.tensor([1e-8, 3.0, -0.0, 1e-30, 65504.])
    ref = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)).float()
    for src in ("f32", "bf16"):
        xs = x if src == "f32" else x.to(torch.bfloat16)
        d = ttnn.from_torch(xs, dtype=ttnn.float32 if src == "f32" else ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        y = ttnn.to_torch(ttnn.typecast(d, ttnn.bfloat8_b)).float()
        diff = (y != ref)
        print(shape, src, "bitid %.6f" % (1 - diff.float().mean().item()), "maxabs %.3e" % (y - ref).abs().max().item(),
              "rel_err_vs_x dev %.3e host %.3e" % ((y - x).norm() / x.norm(), (ref - x).norm() / x.norm()), flush=True)
# timing, 180 Melem in 140 tensors ~ 1.3 Melem each
xs = [torch.randn(1024, 1280, generator=g) for _ in range(40)]
for arm in ("host", "dev_f32", "dev_bf16"):
    t0 = time.time()
    for x in xs:
        if arm == "host":
            t = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        elif arm == "dev_f32":
            t = ttnn.typecast(ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev), ttnn.bfloat8_b)
        else:
            t = ttnn.typecast(ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev), ttnn.bfloat8_b)
    ttnn.synchronize_device(dev)
    dt = time.time() - t0
    print(arm, "%.3f s for %.1f Melem = %.0f Melem/s" % (dt, 40 * 1024 * 1280 / 1e6, 40 * 1024 * 1280 / 1e6 / dt), flush=True)
