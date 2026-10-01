"""rung.py with one measurement-only shim, so 768 can be timed at all on this tree.

At a 768-token seam axis both arms refuse in `autograd.bmm`: the dV-style product
[2,8,768,768]^T @ [2,8,768,32] gets per_core_M=24 and in0_block_w=8 (Kt=24 divides by 8), and
its CBs grow to 1864192 B on a 1572864 B core. 800 (Kt=25 -> 5) and 864 (27 -> 3) fit. This caps
per_core_M * in0_block_w at 128 tiles (16 x 8 at 512 runs), identically in both arms. It is not
a fix and nothing imports it; the fix belongs in `bmm_program_config`.
"""
import runpy, sys
import ttnn
import tt_bio.autograd as ag

_orig = ag.bmm_program_config
CAP = 128


def _capped(a, b, transpose_a=False, transpose_b=False):
    pc = _orig(a, b, transpose_a, transpose_b)
    if pc is None or pc.per_core_M * pc.in0_block_w <= CAP:
        return pc
    sa = [int(d) for d in a.shape]
    Kt = -(-(sa[-2] if transpose_a else sa[-1]) // ttnn.TILE_SIZE)
    w = max(d for d in range(1, pc.in0_block_w + 1) if Kt % d == 0 and pc.per_core_M * d <= CAP)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=pc.compute_with_storage_grid_size, in0_block_w=w,
        out_subblock_h=pc.out_subblock_h, out_subblock_w=pc.out_subblock_w,
        per_core_M=pc.per_core_M, per_core_N=pc.per_core_N)


ag.bmm_program_config = _capped
sys.argv = ["rung.py"] + sys.argv[1:]
runpy.run_path("perf/bgx_size/rung.py", run_name="__main__")
