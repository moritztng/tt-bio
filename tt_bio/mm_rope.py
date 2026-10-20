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


def in1_cores(m: int, n: int, grid) -> int:
    """The cores along minimal_matmul's in1 (column) axis: the grid's y when M > N transposes it, else its x. `permute`
    must use this count, so a weight permuted for one orientation serves only that one."""
    return grid[1] if m > n else grid[0]


def plan(x, w, cfg, grid, seq_len, ncols=None):
    """None, or why the fused kernel declines. `ncols`: the in1 core count the weight was permuted for."""
    m = 1
    for d in list(x.padded_shape)[:-1]:
        m *= int(d)
    n = int(w.padded_shape[-1])
    c = in1_cores(m, n, grid)
    if ncols is not None and ncols != c:
        return f"the weight is permuted for {ncols} in1 cores, this shape runs {c}"
    if seq_len % 32 or int(x.padded_shape[-2]) != seq_len:
        return "rows are not whole sequences of tiles"
    if layout(n // 32, c, cfg[2]) is None:
        return f"qkv {n // 32} tiles do not split into whole heads over {c} cores"
    return None


def qkv_rope_heads(x, w, b, cos, sin, n_heads: int, cfg, grid, ckc, half: bool = False):
    """(rope(q), rope(k), v), each [B, H, L, 64], from x [B, L, D] and the `permute`d weight / bias; cos / sin are the
    [1, 1, L, 64] tables. `half`: the weight is `half_pairs` (then `permute`d), the tables `half_pair_tables`, and the
    heads come out [B, H, L, 32]. None where `plan` declines (the caller keeps the unpermuted weight for that)."""
    seq_len = int(x.padded_shape[-2])
    why = plan(x, w, cfg, grid, seq_len)
    if why is not None:
        STATS[1] += 1
        return None
    dev = x.device()
    bsz = 1
    for d in list(x.padded_shape)[:-2]:
        bsz *= int(d)
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([bsz, n_heads, seq_len, 32 if half else 64]), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    t, p, v0 = layout(int(w.padded_shape[-1]) // 32, in1_cores(bsz * seq_len, int(w.padded_shape[-1]), grid), cfg[2])
    spec = lambda u: (str(u.padded_shape), str(u.dtype), str(u.memory_config()))
    ck = MG.ckc_args(ckc)
    key = (spec(x), spec(w), spec(b), spec(cos), tuple(cfg), tuple(grid), tuple(str(c) for c in ck), half)
    entry = _CACHE.get(key)
    rope = {"MM_HEADS_ST": seq_len // 32, "MM_ROPE_T": t, "MM_ROPE_P": p, "MM_ROPE_V0": v0, **({"MM_HALF_PAIRS": 1} if half else {})}
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


# Heads narrower than 32 (SaProt 35M: head_dim 24). Two heads share a pair of column tiles, A = [h.x1 0.. | h+1.x1 0..]
# and B = [h.x2 0.. | h+1.x2 0..] (x1 / x2 the NeoX halves, each zero-padded to 16 columns), so rotate_half is the tile
# swap the epilogue already does for a 64-wide head. The writer then splits each rotated tile by faces: faces 0 / 2 are
# columns 0-15, faces 1 / 3 columns 16-31, so head h is [A[:, :16] | B[:, :16]] and head h + 1 is [A[:, 16:] | B[:, 16:]],
# each a [L, 32] head [x1 0.. | x2 0..]. q . k is unchanged by the dim order q and k share; v keeps `_pad_heads` order.

def half_pairs(w: torch.Tensor, b: torch.Tensor, n_heads: int, d: int):
    """torch Linear qkv `w` [3 H d, D] and `b` [3 H d] as the [D, 3 H 32] weight and [1, 3 H 32] bias of the half-pair
    layout: q and k as H / 2 tile pairs, v as H zero-padded 32-column heads."""
    assert n_heads % 2 == 0 and d % 2 == 0 and d // 2 <= 16, (n_heads, d)
    h = d // 2
    ws, bs = w.reshape(3, n_heads, d, -1), b.reshape(3, n_heads, d)
    wo = w.new_zeros(3, n_heads, 32, w.shape[-1])
    bo = b.new_zeros(3, n_heads, 32)
    for j in range(2):          # q, k: pair p's tile A is virtual head 2p, tile B is 2p + 1
        for hd in range(n_heads):
            p, s = divmod(hd, 2)
            for half in range(2):
                wo[j, 2 * p + half, 16 * s:16 * s + h] = ws[j, hd, half * h:(half + 1) * h]
                bo[j, 2 * p + half, 16 * s:16 * s + h] = bs[j, hd, half * h:(half + 1) * h]
    wo[2, :, :d], bo[2, :, :d] = ws[2], bs[2]
    return wo.reshape(3 * n_heads * 32, -1).t().contiguous(), bo.reshape(1, -1)


def half_pair_tables(seq_len: int, d: int, base: float = 10000.0, dtype=torch.float32):
    """[L, 64] NeoX cos / sin for the half-pair layout: column c carries frequency c % 16 (< d / 2), both column tiles
    equal (read_rope_block reads one); pad columns hold cos 1, sin 0 over a zero activation."""
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=dtype) / d))
    f = torch.outer(torch.arange(seq_len, dtype=dtype), inv)
    cos, sin = torch.ones(seq_len, 16, dtype=dtype), torch.zeros(seq_len, 16, dtype=dtype)
    cos[:, :d // 2], sin[:, :d // 2] = f.cos(), f.sin()
    return cos.repeat(1, 4), sin.repeat(1, 4)


def half_pair_heads(y: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, n_heads: int):
    """Reference for the fused kernel: [.., L, 3 H 32] half-pair projection -> (q, k, v) [.., H, L, 32] after the
    epilogue's tile-pair RoPE and the writer's face split."""
    t = y.unflatten(-1, (3, n_heads // 2, 2, 32))
    outs = []
    for j in range(2):
        a, b_ = t[..., j, :, 0, :], t[..., j, :, 1, :]                    # [.., L, H / 2, 32]
        c, s = cos[:, None, :32], sin[:, None, :32]
        a, b_ = a * c - b_ * s, b_ * c + a * s
        heads = torch.stack((torch.cat((a[..., :16], b_[..., :16]), -1), torch.cat((a[..., 16:], b_[..., 16:]), -1)), -2)
        outs.append(heads.flatten(-3, -2).movedim(-2, -3))                  # [.., H, L, 32]
    outs.append(y[..., 2 * n_heads * 32:].unflatten(-1, (n_heads, 32)).movedim(-2, -3))
    return tuple(outs)
