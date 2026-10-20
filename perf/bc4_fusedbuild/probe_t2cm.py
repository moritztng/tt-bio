from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, bindcraft2, bmm_res
from tt_bio.af2 import compute_kernel_config
from perf.bc4_fusedbuild.common import N, C, inputs
dev = T.get_device()
z, mask, w = inputs()
sd = {k: v.float() for k, v in w.items()}
Z = ttnn.from_torch(z.reshape(1, N, N, C).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
MK = ttnn.from_torch(mask.reshape(1, N, N).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
with bindcraft2.fast_round():
    m = T.TriangleMultiplication(False, sd, compute_kernel_config())
    o = m(Z, MK, fused_residual=(None,))
    print("took", m.took_residual, "T2CM", TT.T2CM_STATS, "REJECTS", TT.REJECTS, "gin", TT.GIN_RAGGED_STATS, "bres", bmm_res.STATS)
    print("levers", sorted(T._LEVERS))
