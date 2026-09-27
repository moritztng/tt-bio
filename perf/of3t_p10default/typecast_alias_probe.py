"""Does ttnn.typecast to the tensor's own dtype alias it, so deallocating the cast frees the input?"""
import torch


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    for d in (ttnn.float32, ttnn.bfloat16):
        g = ttnn.from_torch(torch.randn(64, 64), dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)
        c = ttnn.typecast(g, d)
        same = c.buffer_address() == g.buffer_address()
        ttnn.deallocate(c)
        print(d, "same buffer", same, "input still allocated", g.is_allocated(), flush=True)


if __name__ == "__main__":
    main()
