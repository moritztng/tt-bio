import torch, ttnn
from tt_bio.tenstorrent import get_device
dev = get_device()
torch.manual_seed(0)
x = torch.randn(288, 8, 2, 2)
ref = torch.softmax(x.double(), -1)
for op in ("softmax_in_place", "softmax"):
    for v in (0.0, -1e30, 50.0, float("nan")):
        t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        t = ttnn.fill_implicit_tile_padding(t, v)
        y = ttnn.softmax_in_place(t) if op == "softmax_in_place" else ttnn.softmax(t, dim=-1)
        o = ttnn.to_torch(y).double()
        print(op, "pad", v, "max|err| vs f64", float((o - ref).abs().max()), "rowsum", float(o.sum(-1).mean()))

