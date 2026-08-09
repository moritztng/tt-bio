#!/usr/bin/env python3
"""Open card 3, report the two numbers every static program-config gate in tt_bio compares against."""
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
print("device id", dev.id())
print("grid", T.COMPUTE_GRID_MAIN, "cores", T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1])
print("get_max_worker_l1_unreserved_size", int(ttnn.get_max_worker_l1_unreserved_size()))
mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
print("L1 banks", mv.num_banks, "total/bank", mv.total_bytes_per_bank,
      "free/bank", mv.total_bytes_free_per_bank,
      "largest_contig_free/bank", mv.largest_contiguous_bytes_free_per_bank)
T.cleanup()
