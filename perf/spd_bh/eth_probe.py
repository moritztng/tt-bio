"""spd-bh: does a Blackhole chip open with Ethernet dispatch, and what compute grid does it give?

tt-bio opens Blackhole with Tensix dispatch (tenstorrent.py, ETH only on single-chip Wormhole), which spends a
Tensix column on dispatch. One process per mode, since a failed ETH open leaves the device half-initialised.
usage: TT_VISIBLE_DEVICES=N timeout 300 python eth_probe.py tensix|eth
"""
import sys, time
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn, torch

mode = sys.argv[1]
kw = {"dispatch_core_config": ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)} if mode == "eth" else {}
t = time.monotonic()
dev = ttnn.open_device(device_id=0, **kw)
g = dev.compute_with_storage_grid_size()
x = ttnn.from_torch(torch.randn(256, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
y = ttnn.to_torch(ttnn.matmul(x, x))
ok = bool(torch.isfinite(y).all())
print(f"RESULT mode={mode} arch={dev.arch()} grid={g.x}x{g.y} cores={g.x * g.y} open_s={time.monotonic() - t:.2f} matmul_finite={ok}")
ttnn.close_device(dev)
