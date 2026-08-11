#!/usr/bin/env python3
"""E6's prize, re-measured on current main rather than inherited from the pre-merge tape.

The 2.871 ms `chunk` and the 2.15 ms input gates come from the predecessor's tape, taken before the
two engine-wide matmul fixes merged at 373d13a3. This times exactly the three ops E6 deletes, at the
production shapes, plus the forward move E6 extends, so the prize is sized against the tree the A/B
will run on.
"""
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.tenstorrent import get_device                                   # noqa: E402
from tt_bio import reblock_permute as RP                                    # noqa: E402

ROOF = 375.0
N, CG = 512, 256                                                            # G=8: 4 roles x 256
dev = get_device()
DRAM = ttnn.DRAM_MEMORY_CONFIG
torch.manual_seed(0)
OUT = {"n": N, "c_group": CG, "roof_gbs": ROOF, "rows": []}


def timed(fn, n=5, warm=2):
    for _ in range(warm):
        for o in fn():
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    keep = [fn() for _ in range(n)]
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3 / n
    for g in keep:
        for o in g:
            ttnn.deallocate(o)
    return ms


def row(label, ms, mb_rw):
    r = {"label": label, "ms": ms, "mb_rw": mb_rw,
         "gbs": mb_rw / 1e3 / (ms * 1e-3), "pct_roof": 100.0 * (mb_rw / 1e3 / (ms * 1e-3)) / ROOF}
    OUT["rows"].append(r)
    print(f"  {label:44s} {ms:7.3f} ms  {r['gbs']:6.1f} GB/s  {r['pct_roof']:5.1f} % of roof",
          flush=True)
    return r


fused = ttnn.from_torch(torch.randn(1, N, N, 4 * CG).bfloat16(), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
mb = N * N * CG * 2 / 1e6                                                   # one role slice, MB

r_chunk = row("ttnn.chunk(gp_in_fused, 4, dim=-1)",
              timed(lambda: list(ttnn.chunk(fused, chunks=4, dim=-1))), 8 * mb)

pieces = list(ttnn.chunk(fused, chunks=4, dim=-1))
g_a, g_b, p_a, p_b = pieces


def gate_once():
    p = ttnn.clone(p_a, memory_config=DRAM)
    return [ttnn.multiply_(p, g_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])]


ms_clone = timed(lambda: [ttnn.clone(p_a, memory_config=DRAM)])
ms_gate_plus_clone = timed(gate_once)
r_gate = row("multiply_(p, g, SIGMOID)  [clone subtracted]", ms_gate_plus_clone - ms_clone, 3 * mb)
row("  the clone that was subtracted", ms_clone, 2 * mb)

a = ttnn.clone(p_a, memory_config=DRAM)
r_move = row("_channel_move forward (reblock_permute)",
             timed(lambda: [RP.reblock_permute(a, memory_config=DRAM, device=dev)]), 2 * mb)

prize = r_chunk["ms"] + 2 * r_gate["ms"]
OUT["prize_ms"] = prize
OUT["move_ms"] = r_move["ms"]
print(f"\n  E6 deletes: chunk {r_chunk['ms']:.3f} + 2 x gate {r_gate['ms']:.3f} = "
      f"{prize:.3f} ms per call", flush=True)
print(f"  E6 extends: the forward move, {r_move['ms']:.3f} ms x 2 calls, at "
      f"{r_move['pct_roof']:.1f} % of the roof", flush=True)
Path(sys.argv[1] if len(sys.argv) > 1 else "e6_prize.json").write_text(json.dumps(OUT, indent=1))
print("wrote", sys.argv[1] if len(sys.argv) > 1 else "e6_prize.json", flush=True)
