"""Does adding a leading unit dim to a [512,512,128] bf16 pair tensor dispatch a program on BH?

The census/trace join charged 0.7292 ms/block to ttnn.reshape(x, (1, *x.shape)) at the end of
TriangleAttention. That join aligned a WH trace onto a BH census by op CLASS, and the rows around
it are a run of consecutive Layout ops, which is exactly where an alignment can be wrong. Ask the
device instead.
"""
import time

import torch

torch.set_grad_enabled(False)
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
if True:
    t_host = torch.randn(512, 512, 128, dtype=torch.float32).bfloat16()
    for space, mc in (("DRAM", ttnn.DRAM_MEMORY_CONFIG), ("L1", ttnn.L1_MEMORY_CONFIG)):
        try:
            t = ttnn.from_torch(t_host, layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=mc)
        except Exception as e:
            print("%s: allocate refused: %s" % (space, str(e)[:140]))
            continue
        cases = (
            ("reshape -> 1x512x512x128", lambda x: ttnn.reshape(x, (1, 512, 512, 128))),
            ("reshape -> 512x512x128", lambda x: ttnn.reshape(x, (512, 512, 128))),
            ("permute (1,0,2)", lambda x: ttnn.permute(x, (1, 0, 2), memory_config=mc)),
        )
        for label, fn in cases:
            ts = []
            alias = None
            for _ in range(4):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = fn(t)
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
                alias = o.buffer_address() == t.buffer_address()
                if not alias:
                    ttnn.deallocate(o)
            print("%-5s %-26s ms %s  aliases_input=%s" % (
                space, label, ["%.4f" % v for v in ts], alias))
        ttnn.deallocate(t)
