"""`inproj_gated.device_weights` (on card, per call) against `prepare_weights` (host): torch.equal."""
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import inproj_gated as IG
from tt_bio.tenstorrent import get_device
dev = get_device()
torch.manual_seed(0)
ok = True
for K, C4 in ((128, 512), (128, 256), (64, 512)):
    for dt in (ttnn.bfloat16,):
        for bias in (False, True):
            w = torch.randn(K, C4)
            b = torch.randn(C4) if bias else None
            ref, _ = IG.prepare_weights(w, b, dev)
            wd = ttnn.from_torch(w.reshape(1, 1, K, C4), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            got = IG.device_weights(wd, None if b is None else IG.bias_column(b, dev))
            a, g = ttnn.to_torch(ref), ttnn.to_torch(got)
            eq = tuple(a.shape) == tuple(g.shape) and torch.equal(a, g)
            ok &= eq
            print(K, C4, dt, "bias" if bias else "nobias", tuple(g.shape), "EQUAL" if eq else "DIFF")
print("ALL EQUAL" if ok else "MISMATCH")
# Why fp32 weights are declined: the device cast against torch's round-to-nearest-even.
w = torch.randn(128, 512)
rne = w.t().contiguous().bfloat16().float()
wd = ttnn.from_torch(w.t().contiguous().reshape(1, 1, 512, 128), dtype=ttnn.float32,
                     layout=ttnn.TILE_LAYOUT, device=dev)
d = ttnn.to_torch(ttnn.typecast(wd, ttnn.bfloat16)).float().reshape(512, 128)
print("ttnn.typecast fp32->bf16 vs torch RNE: mismatch fraction", (d != rne).float().mean().item())
