"""ag.softmax_bw with the fp32 backward off, bf16 y and fp32 g, fresh tensors, no prior call."""
import torch


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    ag.SOFTMAX_BW_FP32 = False
    torch.manual_seed(0)
    shape = [12, 4, 128, 128]
    tt = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)
    y = tt(torch.softmax(torch.randn(shape) * 3, -1), ttnn.bfloat16)
    g = tt(torch.randn(shape), ttnn.float32)
    y64, g64 = ttnn.to_torch(y).double(), ttnn.to_torch(g).double()
    ref = y64 * (g64 - (g64 * y64).sum(-1, keepdim=True) / y64.sum(-1, keepdim=True))
    for cfg in (None, ag.precise_config()):
        for _ in range(2):
            dx = ttnn.to_torch(ag.softmax_bw(y, g, dim=-1, config=cfg)).double()
            print("config", cfg is not None, "rel", float((dx - ref).norm() / ref.norm()), flush=True)
    inner = ag.softmax_bw_inner(y, g, dim=-1, config=ag.precise_config())
    i_ref = (g64 * y64).sum(-1, keepdim=True) / y64.sum(-1, keepdim=True)
    print("inner rel", float((ttnn.to_torch(inner).double() - i_ref).norm() / i_ref.norm()))
    sub = ttnn.subtract(g, inner)
    print("sub rel", float((ttnn.to_torch(sub).double() - (g64 - ttnn.to_torch(inner).double())).norm() / g64.norm()), sub.dtype)
    out = ttnn.multiply(y, sub)
    print("out rel", float((ttnn.to_torch(out).double() - ref).norm() / ref.norm()), out.dtype)


if __name__ == "__main__":
    main()
