import sys
import ttnn, torch
from tt_bio.tenstorrent import get_device, cleanup

d = get_device()
a = ttnn.from_torch(torch.ones(64, 64), layout=ttnn.TILE_LAYOUT, device=d, dtype=ttnn.bfloat16)
got, want = ttnn.to_torch(ttnn.matmul(a, a)).sum().item(), 64 ** 3
cleanup()
if abs(got - want) >= 1:
    sys.exit(f"card computes wrong: {got} != {want}")
