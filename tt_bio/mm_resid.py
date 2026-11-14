"""`resid + (x @ w + bias)` as ONE minimal_matmul (kernels/mm_gate under MM_RESID): the transformer block's residual add
in the matmul's epilogue, in place of the projection and a separate ttnn.add.

The non-writing DM kernel reads the residual's output block into c_5 beside the bias; the compute kernel adds the bias
as the stock epilogue does, then the residual tile into dest on the FPU, and packs once. The weight may be block-fp8
(the wheel's fused dit_minimal_matmul_addcmul_fused needs bf16, and a bf16 copy of --fast's weights cost more than the
add it saved). Everything else is the wheel's minimal_matmul through `mm_generic.build`.
"""
import ttnn

from tt_bio import mm_generic as MG
from tt_bio.mm_gate import KERNEL_DIR

STATS = [0, 0]   # runs, declines
_CACHE: dict = {}


def matmul_add(x, w, b, resid, cfg, grid, ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """resid + x @ w + b, [.., N] bf16, for block config `cfg` (M, K, N, subblock_h, subblock_w) on `grid`, or None where
    the residual's tile grid is not the output's (the caller keeps the projection and the add)."""
    n = int(w.padded_shape[-1])
    if list(resid.padded_shape)[:-1] != list(x.padded_shape)[:-1] or int(resid.padded_shape[-1]) != n \
            or resid.layout != ttnn.TILE_LAYOUT or b is None:
        STATS[1] += 1
        return None
    dev = x.device()
    shape = [int(d) for d in x.shape][:-1] + [int(w.shape[-1])]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, memory_config)
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    ck = MG.ckc_args(ckc)
    key = (spec(x), spec(w), spec(b), spec(resid), spec(out), tuple(cfg), tuple(grid), tuple(str(c) for c in ck))
    entry = _CACHE.get(key)
    if entry is None:
        entry = MG.build(dev, x, w, [out], (tuple(cfg), tuple(grid)), ck, defines={"MM_RESID": 1}, kernel_dir=KERNEL_DIR,
                         bias=b, extra=(resid,))
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])
        entry["cbs"].append(MG._cb(5, core_grid, MG.tile_bytes(resid.dtype), cfg[0] * cfg[2], resid.dtype))
        compute = entry["kernels"][4]
        compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
        compute.defines = [("FUSE_BIAS", "1"), ("MM_RESID", "1")]
        entry["pd"] = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["semaphores"], cbs=entry["cbs"])
        entry["extra"] = (resid.buffer_address(),)
        _CACHE[key] = entry
    addrs = (x.buffer_address(), w.buffer_address(), (out.buffer_address(),), b.buffer_address())
    extra = (resid.buffer_address(),)
    if addrs != entry["addrs"] or extra != entry["extra"]:
        MG.rebind(entry, *addrs, extra_addrs=extra)
        entry["extra"] = extra
    ttnn.generic_op([x, w, resid, out], entry["pd"])
    STATS[0] += 1
    return out
