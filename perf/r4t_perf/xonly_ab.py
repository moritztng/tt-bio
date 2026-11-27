"""x-only lnbw (the BindCraft 2 / inference-path form) must be byte-identical before and after the L2 kernel change."""
import hashlib, sys, torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio import tenstorrent as T, lnbw
dev = T.get_device()
lnbw.FUSED = True
gen = torch.Generator().manual_seed(3)
put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
h = hashlib.sha256()
for shp in ([1, 256, 256, 256], [1, 288, 288, 128]):
    x, g = torch.randn(*shp, generator=gen), torch.randn(*shp, generator=gen)
    gm = 1 + 0.3 * torch.randn(1, shp[-1], generator=gen)
    for gamma in (None, put(gm)):
        dx = lnbw.layer_norm_bw(put(x), put(g), gamma, 1e-5)
        h.update(ttnn.to_torch(dx).float().numpy().tobytes())
print("XONLY", lnbw.__file__, h.hexdigest()[:16])
