"""SaProt's qkv projection, bias, NeoX RoPE and head split as one minimal_matmul (kernels/mm_gate under MM_ROPE_T):
q, k, v come out [B, H, L, 64] in the layout SDPA reads, in place of minimal_matmul + qk_rope_heads.rope_heads_packed.

The epilogue rotates a head from the accumulator only if both its tiles sit in one core's output block, so the qkv
weight's columns are permuted per in1-axis core: each core's T = 3 KT / ncols tiles are P = 2 KT / ncols q / k tiles
(whole heads) then the core's share of v (`permute`). The writer maps every tile back to its q, k or v head
(matmul_dataflow_common.hpp rope_col, chunk_tile_id). `plan` declines what that layout cannot carry.
"""
import torch
import ttnn

from tt_bio import mm_generic as MG
from tt_bio.mm_gate import KERNEL_DIR

STATS = [0, 0]   # runs, declines
_CACHE: dict = {}


def layout(n_tiles: int, ncols: int, n_block: int):
    """(T, P, V0) for a [.., 3 KT] qkv over `ncols` in1-axis cores, or None: every core needs whole q / k heads and
    an equal share of v, and an even N block so no head straddles two blocks."""
    kt = n_tiles // 3
    if n_tiles % 3 or n_tiles % ncols or (2 * kt) % (2 * ncols) or kt % ncols or n_block % 2:
        return None
    return n_tiles // ncols, 2 * kt // ncols, 2 * kt


def permute(w: torch.Tensor, ncols: int, n_block: int) -> torch.Tensor:
    """The [.., 3 D] qkv weight (or [1, 3 D] bias) with its 32-column tiles in the kernel's per-core order."""
    n_tiles = w.shape[-1] // 32
    t, p, v0 = layout(n_tiles, ncols, n_block)

    def col(n):
        r, j = divmod(n, t)
        return r * p + j if j < p else v0 + r * (t - p) + j - p
    idx = torch.tensor([col(n) * 32 + i for n in range(n_tiles) for i in range(32)])
    return w[..., idx]


def plan(x, w, cfg, grid, seq_len):
    """None, or why the fused kernel declines."""
    m = 1
    for d in list(x.padded_shape)[:-1]:
        m *= int(d)
    n = int(w.padded_shape[-1])
    if m > n:
        return "M > N transposes the core grid; the weight is permuted for its columns"
    if seq_len % 32 or int(x.padded_shape[-2]) != seq_len:
        return "rows are not whole sequences of tiles"
    if layout(n // 32, grid[0], cfg[2]) is None:
        return f"qkv {n // 32} tiles do not split into whole heads over {grid[0]} cores"
    return None


def qkv_rope_heads(x, w, b, cos, sin, n_heads: int, cfg, grid, ckc):
    """(rope(q), rope(k), v), each [B, H, L, 64], from x [B, L, D] and the `permute`d weight / bias; cos / sin are the
    [1, 1, L, 64] tables. None where `plan` declines (the caller keeps the unpermuted weight for that)."""
    seq_len = int(x.padded_shape[-2])
    why = plan(x, w, cfg, grid, seq_len)
    if why is not None:
        STATS[1] += 1
        return None
    dev = x.device()
    bsz = 1
    for d in list(x.padded_shape)[:-2]:
        bsz *= int(d)
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([bsz, n_heads, seq_len, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                           dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    t, p, v0 = layout(int(w.padded_shape[-1]) // 32, grid[0], cfg[2])
    spec = lambda u: (str(u.padded_shape), str(u.dtype), str(u.memory_config()))
    ck = MG.ckc_args(ckc)
    key = (spec(x), spec(w), spec(b), spec(cos), tuple(cfg), tuple(grid), tuple(str(c) for c in ck))
    entry = _CACHE.get(key)
    rope = {"MM_HEADS_ST": seq_len // 32, "MM_ROPE_T": t, "MM_ROPE_P": p, "MM_ROPE_V0": v0}
    if entry is None:
        entry = MG.build(dev, x, w, outs, (tuple(cfg), tuple(grid)), ck, defines=rope, kernel_dir=KERNEL_DIR, bias=b,
                         extra=(cos, sin))
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])
        ts = MG.tile_bytes(cos.dtype)
        entry["cbs"] += [MG._cb(7, core_grid, ts, cfg[0] * 2, cos.dtype), MG._cb(8, core_grid, ts, cfg[0] * 2, sin.dtype),
                         MG._cb(9, core_grid, MG.tile_bytes(ttnn.bfloat16), cfg[2], ttnn.bfloat16)]
        compute = entry["kernels"][4]
        compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
        compute.defines = [("FUSE_BIAS", "1")] + [(k, str(v)) for k, v in rope.items()]
        entry["pd"] = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["semaphores"], cbs=entry["cbs"])
        entry["extra"] = (cos.buffer_address(), sin.buffer_address())
        _CACHE[key] = entry
    addrs = (x.buffer_address(), w.buffer_address(), tuple(o.buffer_address() for o in outs), b.buffer_address())
    extra = (cos.buffer_address(), sin.buffer_address())
    if addrs != entry["addrs"] or extra != entry["extra"]:
        MG.rebind(entry, *addrs, extra_addrs=extra)
        entry["extra"] = extra
    ttnn.generic_op([x, w, *outs], entry["pd"])
    STATS[0] += 1
    return tuple(outs)
