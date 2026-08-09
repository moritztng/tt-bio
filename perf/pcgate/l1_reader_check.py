#!/usr/bin/env python3
"""Instrument check: does largest_contiguous_bytes_free_per_bank actually respond to L1 use?

The gate probe reports the full 1461760 B/bank free at every gated call in a 298 aa fold. That is
only meaningful if the reader moves when L1 IS occupied, so allocate an L1 tensor and watch it.
"""
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()


def free():
    return int(ttnn.get_memory_view(dev, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank)


print("idle                     ", free())
a = ttnn.from_torch(torch.zeros(1, 320, 320, 256, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
print("after 50.3 MB L1 tensor  ", free())
b = ttnn.from_torch(torch.zeros(1, 320, 320, 256, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
print("after a second one       ", free())
ttnn.deallocate(a)
ttnn.deallocate(b)
print("after deallocate         ", free())
T.cleanup()
