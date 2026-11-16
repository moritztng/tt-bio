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


def rowstats_blocks(n: int, cfg) -> int:
    """Stats blocks along N of a `matmul_add(..., rowstats=True)` over n columns: one per N block holding real columns."""
    return -(-(n // 32) // cfg[2])


def matmul_add(x, w, b, resid, cfg, grid, ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, rowstats=False):
    """resid + x @ w + b, [.., N] bf16, for block config `cfg` (M, K, N, subblock_h, subblock_w) on `grid`, or None where
    the residual's tile grid is not the output's (the caller keeps the projection and the add).
    `rowstats`: also the next LayerNorm's row statistics of that bf16 output, returned as (out, stats). stats is fp32
    [rows, 64 * rowstats_blocks]: per N block g, tile column 2 g holds the row's sum of x over the block and 2 g + 1
    its sum of x^2 (x^2 rounded to bf16 first), every column of a tile the same value. Summing the blocks gives the
    row's sum x and sum x^2 for rstd = rsqrt(E x^2 - (E x)^2 + eps); perf/spd_m_esmc/lnacc.py grades it (`pc`)."""
    n = int(w.padded_shape[-1])
    if list(resid.padded_shape)[:-1] != list(x.padded_shape)[:-1] or int(resid.padded_shape[-1]) != n \
            or resid.layout != ttnn.TILE_LAYOUT or b is None:
        STATS[1] += 1
        return None
    dev = x.device()
    shape = [int(d) for d in x.shape][:-1] + [int(w.shape[-1])]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, memory_config)
    tensors = (resid,)
    defines = {"MM_RESID": 1}
    if rowstats:
        nt, nb = n // 32, rowstats_blocks(n, cfg)
        rows = x.volume() // int(x.shape[-1])
        tensors += (ttnn.allocate_tensor_on_device(ttnn.Shape([rows, 64 * nb]), ttnn.float32, ttnn.TILE_LAYOUT, dev,
                                                   ttnn.DRAM_MEMORY_CONFIG),)
        defines.update(MM_ROWSTATS=1, MM_ROWSTATS_N=nt, MM_ROWSTATS_NB=nb)
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    ck = MG.ckc_args(ckc)
    key = (spec(x), spec(w), spec(b), spec(resid), spec(out), tuple(cfg), tuple(grid), tuple(str(c) for c in ck),
           rowstats)
    entry = _CACHE.get(key)
    if entry is None:
        # rowstats: every core's N range whole N blocks (n_align), so a block's index along N is n_tile / N_block
        entry = MG.build(dev, x, w, [out], (tuple(cfg), tuple(grid)), ck, defines=defines, kernel_dir=KERNEL_DIR,
                         bias=b, extra=tensors, **({"n_align": cfg[2]} if rowstats else {}))
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])
        entry["cbs"].append(MG._cb(5, core_grid, MG.tile_bytes(resid.dtype), cfg[0] * cfg[2], resid.dtype))
        if rowstats:   # compute.cpp rowstats_row: bf16 x and x^2 of one row, fp32 [sum x, sum x^2] per row, a ones tile
            bf, f32 = MG.tile_bytes(ttnn.bfloat16), MG.tile_bytes(ttnn.float32)
            entry["cbs"] += [MG._cb(10, core_grid, bf, cfg[2], ttnn.bfloat16), MG._cb(11, core_grid, bf, cfg[2], ttnn.bfloat16),
                             MG._cb(12, core_grid, f32, 2 * cfg[0], ttnn.float32), MG._cb(13, core_grid, bf, 1, ttnn.bfloat16)]
        compute = entry["kernels"][4]
        compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
        compute.defines = [("FUSE_BIAS", "1")] + [(k, str(v)) for k, v in defines.items()]
        entry["pd"] = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["semaphores"], cbs=entry["cbs"])
        entry["extra"] = tuple(t.buffer_address() for t in tensors)
        _CACHE[key] = entry
    addrs = (x.buffer_address(), w.buffer_address(), (out.buffer_address(),), b.buffer_address())
    extra = tuple(t.buffer_address() for t in tensors)
    if addrs != entry["addrs"] or extra != entry["extra"]:
        MG.rebind(entry, *addrs, extra_addrs=extra)
        entry["extra"] = extra
    ttnn.generic_op([x, w, *tensors, out], entry["pd"])
    STATS[0] += 1
    return (out, tensors[1]) if rowstats else out
