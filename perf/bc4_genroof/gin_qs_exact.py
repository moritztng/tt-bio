"""torch.equal of gin_ragged's outputs at qs 1 and the auto qs, at BC2's two 224 shapes, masked.

usage: TT_VISIBLE_DEVICES=<chip> python perf/bc4_genroof/gin_qs_exact.py
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
grid = tuple(T.COMPUTE_GRID_MAIN)
DR = ttnn.DRAM_MEMORY_CONFIG
CK = MG.ckc_args(compute_kernel_config())
up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                               device=dev, memory_config=DR)
n, K = 224, 128
for C in (128, 64):
    g = torch.Generator().manual_seed(C)
    x = torch.randn(1, n, n, K, generator=g).to(torch.bfloat16)
    wp, wg = ((torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16) for _ in range(2))
    bp, bg = ((torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16) for _ in range(2))
    m = (torch.rand(1, n, n, generator=g) > 0.1).to(torch.bfloat16)
    args = (up(x), up(wp.t()), up(wg.t()), TT.gin_ragged_bias(bp, bg, n, dev), CK, grid, n)
    res = {}
    for qs in (1, 0):
        TT.GIN_RAGGED_QS = qs
        res[qs] = [ttnn.to_torch(t) for t in TT.gin_ragged(*args, mask=up(m), memory_config=DR)]
    auto = TT._gin_ragged_qs(2 * C // 32, (-(-n // 32)) ** 2, grid[0] * grid[1])
    print(json.dumps({"c": C, "grid": grid, "auto_qs": auto,
                      "equal": all(torch.equal(a, b) for a, b in zip(res[1], res[0]))}), flush=True)
