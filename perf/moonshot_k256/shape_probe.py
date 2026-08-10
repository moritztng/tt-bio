#!/usr/bin/env python3
"""What shape does the trimul input projection ACTUALLY execute at on this host?

`ceiling-298aa.md` censused the fold on qb2 (11x10 = 110 cores) and recorded the trimul input
projection as `[1,320,320,256] @ [256,128]`. That N=128 is not a property of the model: the
projection is already fused over `[g_a | g_b | p_a | p_b]` (`_gp_in_chunks`), so its N is exactly
`4 * _trimul_chunk_size(...)`, and `_trimul_chunk_size` doubles the chunk while the chunk's L1
working set fits a budget that scales with the core grid. A 130-core host therefore executes a
different shape than the census recorded, on the single largest arithmetic class in the fold.

Prints the executed N for both grids at the sizes this leg quotes, so nobody has to infer it
from a FLOP count again.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:moonshot-4x-k256-kernel-rate \
        python3 perf/moonshot_k256/shape_probe.py out.json
"""
import json
import sys

import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
print(f"device grid {g.x}x{g.y} = {g.x * g.y} cores; tt_bio COMPUTE_GRID_MAIN {T.COMPUTE_GRID_MAIN}")
print(f"L1 unreserved per core {ttnn.get_max_worker_l1_unreserved_size()}")
print(f"_trimul_l1_max_seq() = {T._trimul_l1_max_seq()} (_FAST_MODE={T._FAST_MODE})")
print(f"TRIANGLE_MULT_L1_CHUNK_BUDGET = {T.TRIANGLE_MULT_L1_CHUNK_BUDGET}")

# protenix-v2 trunk trimul c_hidden, read off protenix-v2.pt: pairformer_stack.blocks.0
# .tri_mul_out.linear_a_p.weight is (256, 256) = [c_hidden, c_z].
HIDDEN = 256
out = {
    "grid": [g.x, g.y],
    "cores": g.x * g.y,
    "hidden": HIDDEN,
    "l1_max_seq": T._trimul_l1_max_seq(),
    "chunk_budget": T.TRIANGLE_MULT_L1_CHUNK_BUDGET,
    "rows": [],
}
for seq in (288, 320, 512, 544):
    C = T._trimul_chunk_size(seq, HIDDEN, 1)
    row = {
        "seq": seq,
        "chunk": C,
        "n_pairs": HIDDEN // C,
        "in_proj_N": 4 * C,
        "result_mem": str(T._triangle_mul_memory_config(seq).buffer_type).split(".")[-1],
    }
    out["rows"].append(row)
    print(
        f"  seq {seq:4d}: chunk {C:3d}  n_pairs {row['n_pairs']}  "
        f"in_proj N = {row['in_proj_N']:4d}  result -> {row['result_mem']}"
    )

if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
