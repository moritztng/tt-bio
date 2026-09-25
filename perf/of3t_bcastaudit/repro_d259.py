"""ttnn.multiply(bf16 x, fp32 b), b broadcast, goes wrong after other multiplies have run.

x is [1, 64, 64, 128] and b a 0/1 mask [1, 64, 64, 1], so x * b is exact in any dtype. Each step
runs 10 reps of multiply(x, b) at the named dtypes; before every rep eight 3e4-filled buffers are
allocated and freed, so a read of device memory the op does not own shows up. Run each arm in a
fresh process:

    python3 repro_d259.py bf16xf32             # 0 wrong on every rep
    python3 repro_d259.py f32xbf16 bf16xf32    # f32xbf16 exact; then bf16xf32 wrong on most reps

Seen on Blackhole p300c, ttnn 0.68.0 (repro_d259.out). A lone p300c chip opens only with
TT_MESH_GRAPH_DESC_PATH set to p150_mesh_graph_descriptor.textproto.
"""
import sys
import torch
import ttnn

dev = ttnn.open_device(device_id=0)
up = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
g = torch.Generator().manual_seed(0)
xt, m = torch.randn(1, 64, 64, 128, generator=g), (torch.rand(1, 64, 64, 1, generator=g) > 0.2).float()
D = {"f32": ttnn.float32, "bf16": ttnn.bfloat16}
for step in sys.argv[1:]:
    xd, md = step.split("x")
    x = up(xt, D[xd])
    ref, wrong = ttnn.to_torch(x).float() * m, []
    for rep in range(10):
        for t in [up(torch.full((1, 64, 64, 128), 3.0e4), ttnn.float32) for _ in range(8)]:
            ttnn.deallocate(t)
        wrong.append(int((ttnn.to_torch(ttnn.multiply(x, up(m, D[md]))).float() != ref).sum()))
    print(step, "wrong elements per rep:", wrong, flush=True)
ttnn.close_device(dev)
