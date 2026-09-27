"""Which verb of the fp32-off softmax backward is wrong on a bf16 x fp32 operand pair?"""
import torch


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    torch.manual_seed(0)
    shape = [12, 4, 128, 128]
    yt = torch.softmax(torch.randn(shape) * 3, -1)
    gt = torch.randn(shape)
    tt = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)
    y = tt(yt, ttnn.bfloat16)
    g = tt(gt, ttnn.float32)
    y64 = ttnn.to_torch(y).double()
    g64 = ttnn.to_torch(g).double()
    rel = lambda a, b: float((ttnn.to_torch(a).double() - b).norm() / b.norm())
    prod = ttnn.multiply(g, y)
    print("multiply(g32, y16)", prod.dtype, rel(prod, g64 * y64))
    prod2 = ttnn.multiply(y, g)
    print("multiply(y16, g32)", prod2.dtype, rel(prod2, g64 * y64))
    inner = ttnn.sum(prod, dim=-1, keepdim=True)
    print("sum", inner.dtype, rel(inner, (g64 * y64).sum(-1, keepdim=True)))
    i64 = ttnn.to_torch(inner).double()
    d = ttnn.subtract(g, inner)
    print("subtract(g32, inner)", d.dtype, rel(d, g64 - i64))
    d64 = ttnn.to_torch(d).double()
    out = ttnn.multiply(y, d)
    print("multiply(y16, d32)", out.dtype, rel(out, y64 * d64))


if __name__ == "__main__" and __import__("sys").argv[-1] != "renorm":
    main()


def renorm():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    torch.manual_seed(0)
    shape = [12, 4, 128, 128]
    yt = torch.softmax(torch.randn(shape) * 3, -1)
    gt = torch.randn(shape)
    tt = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)
    y, g = tt(yt, ttnn.bfloat16), tt(gt, ttnn.float32)
    y64, g64 = ttnn.to_torch(y).double(), ttnn.to_torch(g).double()
    rel = lambda a, b: float((ttnn.to_torch(a).double() - b).norm() / b.norm())
    inner = ttnn.sum(ttnn.multiply(g, y), dim=-1, keepdim=True)
    i64 = ttnn.to_torch(inner).double()
    s = ttnn.sum(y, dim=-1, keepdim=True, compute_kernel_config=ag.precise_config())
    s64 = ttnn.to_torch(s).double()
    print("sum(y16)", s.dtype, rel(s, y64.sum(-1, keepdim=True)))
    q = ttnn.divide(inner, s)
    print("divide(inner32, s16)", q.dtype, rel(q, i64 / s64))
    q2 = ttnn.divide(inner, ttnn.typecast(s, ttnn.float32))
    print("divide(inner32, s32)", q2.dtype, rel(q2, i64 / s64))


if __name__ == "__main__" and __import__("sys").argv[-1] == "renorm":
    renorm()
