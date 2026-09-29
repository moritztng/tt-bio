"""Is there an EXACT way to move a channel axis off the tiled pair, and what does it cost?

The point term needs (i, j) on the two tiled axes; the projection that produces the points puts
(residue, channel) there. One transpose is structurally required, and the tile transpose rounds
fp32 at 4.95e-04, which on a 40 A global coordinate is 0.02 A and lands back in the same
cancellation the matmul identity died of.
"""
import time, torch, ttnn
from tt_bio.tenstorrent import get_device
dev = get_device()
F32 = ttnn.float32
B, C, N = 8, 144, 256
x = torch.randn(B, C, N, dtype=torch.float64)
def tt(t): return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=F32)
def err(got, ref): 
    s = max(ref.abs().max().item(), 1e-30)
    return (got.double()-ref.double()).abs().max().item()/s

ref = x.transpose(-1, -2)
for name, fn in (
    ("tile permute (0,2,1)", lambda t: ttnn.permute(t, (0, 2, 1))),
    ("tile transpose", lambda t: ttnn.transpose(t, -2, -1)),
    ("row-major round trip, no move",
     lambda t: ttnn.to_layout(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT)),
    ("row-major permute then retile",
     lambda t: ttnn.to_layout(ttnn.permute(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), (0, 2, 1)),
                              ttnn.TILE_LAYOUT)),
):
    try:
        t0 = time.perf_counter()
        out = ttnn.to_torch(fn(tt(x)))
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3
        target = x if "no move" in name else ref
        print(f"  {'ok  ' if err(out, target) < 1e-6 else 'ROUNDS'} {name:<32}"
              f" rel {err(out, target):.2e}  {ms:6.1f} ms incl. host copy")
    except Exception as exc:
        print(f"  FAIL {name:<32} {repr(exc)[:100]}")

print("\nand the shape the point term actually wants: [B,C,N] -> [B*C,N,1] and [B*C,1,N]")
for name, shape in (("[B,C,N,1] via reshape", [B, C, N, 1]), ("[B,C,1,N] via reshape", [B, C, 1, N])):
    try:
        out = ttnn.to_torch(ttnn.reshape(tt(x), shape)).reshape(B, C, N)
        print(f"  {'ok  ' if err(out, x) < 1e-6 else 'ROUNDS'} {name:<32} rel {err(out, x):.2e}")
    except Exception as exc:
        print(f"  FAIL {name:<32} {repr(exc)[:100]}")
ttnn.close_device(dev)
