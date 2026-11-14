"""Device precision probe (qb2 p300c, chip pinned via TT_VISIBLE_DEVICES; see state/rfd4-vae.md for the readings)."""
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
up = lambda t: ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
a = torch.randn(1, 256, 768, dtype=torch.float64).float().double(); b = torch.randn(1, 256, 768, dtype=torch.float64).float().double()
A, B = up(a), up(b)
def rel(y, r): return float(((ttnn.to_torch(y).double() - r).abs() / r.abs().clamp_min(1e-3)).max())
for leg in (None, True, False):
    kw = {} if leg is None else {"use_legacy": leg}
    try:
        print("legacy", leg, "mul", rel(ttnn.multiply(A, B, **kw), a * b), "add", rel(ttnn.add(A, B, **kw), a + b), "sub", rel(ttnn.subtract(A, B, **kw), a - b))
    except Exception as e:
        print("legacy", leg, "err", str(e)[:120])
v = a.abs() + 0.1
print("rsqrt", rel(ttnn.rsqrt(up(v)), v.rsqrt()), "sqrt", rel(ttnn.sqrt(up(v)), v.sqrt()), "recip", rel(ttnn.reciprocal(up(v)), 1 / v))
