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
_L1_REFUSED: set = set()   # head shapes whose L1 q / k the device refused (allocation or a CB clash)
PROBE_DEFINES: dict = {}   # benchmarks only (perf/spd_m_esmc/ropeceil.py): extra compute defines, e.g. MM_ROPE_STATS_PROBE


def layout(n_tiles: int, ncols: int, n_block: int):
    """(T, P, V0) for a [.., 3 KT] qkv over `ncols` in1-axis cores, or None: every core needs whole q / k heads and
    an equal share of v, and an even N block so no head straddles two blocks."""
    kt = n_tiles // 3
    if n_tiles % 3 or n_tiles % ncols or (2 * kt) % (2 * ncols) or kt % ncols or n_block % 2:
        return None
    return n_tiles // ncols, 2 * kt // ncols, 2 * kt


def cols(n_tiles: int, ncols: int, n_block: int):
    """The widest in1 core count <= `ncols` that `layout` accepts, or None. Blackhole's 11 / 13 columns split no qkv of
    whole 64-dim heads (650M: 120 tiles), so it runs 10 and leaves a column idle; Wormhole's 8 is kept."""
    return next((c for c in range(ncols, 0, -1) if layout(n_tiles, c, n_block) is not None), None)


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
    if ncols is None and m > n:   # a caller that names no ncols permuted for grid x, the untransposed axis
        return "M > N transposes the core grid; the weight is permuted for its columns"
    if ncols is not None and ncols != c:
        return f"the weight is permuted for {ncols} in1 cores, this shape runs {c}"
    if seq_len % 32 or int(x.padded_shape[-2]) != seq_len:
        return "rows are not whole sequences of tiles"
    if layout(n // 32, c, cfg[2]) is None:
        return f"qkv {n // 32} tiles do not split into whole heads over {c} cores"
    return None


def qkv_rope_heads(x, w, b, cos, sin, n_heads: int, cfg, grid, ckc, half: bool = False, ncols=None, qk_l1: bool = False,
                   qk_norm: bool = False):
    """(rope(q), rope(k), v), each [B, H, L, 64], from x [B, L, D] and the `permute`d weight / bias; cos / sin are the
    [1, 1, L, 64] tables. `half`: the weight is `half_pairs` (then `permute`d), the tables `half_pair_tables`, and the
    heads come out [B, H, L, 32] (`w` may carry zero pad heads past H, which are not written). `ncols`: the in1 core count `w` was permuted for (needed when M > N). `qk_l1`: q and k go to L1, where SDPA reads them (DRAM for a shape whose
    allocation L1 refuses, from then on). None where `plan` declines (the caller keeps the unpermuted weight for that).
    `qk_norm`: ESMC's q / k LayerNorm, from `qk_norm_fold`'s weight and [bias; g] `b`. q and k come out g-scaled and
    rotated but not yet divided by their row's sd, and a fourth tensor, the fp32 [rows, 64 NB] partial sums of the
    centred q^2 / k^2 (`qk_rstd` turns them into the rows' rstd)."""
    seq_len = int(x.padded_shape[-2])
    why = plan(x, w, cfg, grid, seq_len, ncols)
    if why is not None:
        STATS[1] += 1
        return None
    dev = x.device()
    bsz = 1
    for d in list(x.padded_shape)[:-2]:
        bsz *= int(d)
    shape = ttnn.Shape([bsz, n_heads, seq_len, 32 if half else 64])
    alloc = lambda mc: ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)
    outs = []
    if qk_l1 and tuple(shape) not in _L1_REFUSED:
        try:
            for _ in range(2):
                outs.append(alloc(ttnn.L1_MEMORY_CONFIG))
        except Exception as exc:
            from tt_bio import esmc, tenstorrent
            if not esmc._is_alloc_refusal(exc):
                tenstorrent.absorb_l1_refusal("mm_rope qk L1", exc)
            _L1_REFUSED.add(tuple(shape))
            for t in outs:
                ttnn.deallocate(t)
            outs = []
    outs += [alloc(ttnn.DRAM_MEMORY_CONFIG) for _ in range(3 - len(outs))]
    t, p, v0 = layout(int(w.padded_shape[-1]) // 32, in1_cores(bsz * seq_len, int(w.padded_shape[-1]), grid), cfg[2])
    extra = (cos, sin)
    qkn = {}
    if qk_norm:
        bpc = -(-t // cfg[2])
        nb = (int(w.padded_shape[-1]) // 32 // t) * bpc
        extra += (ttnn.allocate_tensor_on_device(ttnn.Shape([bsz * seq_len, 64 * nb]), ttnn.float32, ttnn.TILE_LAYOUT,
                                                 dev, ttnn.DRAM_MEMORY_CONFIG),)
        qkn = {"MM_ROWSTATS": 1, "MM_ROWSTATS_N": 1 << 30, "MM_ROWSTATS_NB": nb, "MM_ROPE_QKN_BPC": bpc,
               "MM_ROPE_QKN_W": int(b.padded_shape[-1]) // 32}
    spec = lambda u: (str(u.padded_shape), str(u.dtype), str(u.memory_config()))
    ck = MG.ckc_args(ckc)
    key = (spec(x), spec(w), spec(b), spec(cos), tuple(cfg), tuple(grid), tuple(str(c) for c in ck), half,
           tuple(sorted(PROBE_DEFINES.items())), qk_norm,
           tuple(o.memory_config().buffer_type for o in outs))
    entry = _CACHE.get(key)
    rope = {"MM_HEADS_ST": seq_len // 32, "MM_ROPE_T": t, "MM_ROPE_P": p, "MM_ROPE_V0": v0,
            **({"MM_HALF_PAIRS": 1, "MM_HEADS_H": n_heads} if half else {}), **qkn, **PROBE_DEFINES}
    if entry is None:
        entry = MG.build(dev, x, w, outs, (tuple(cfg), tuple(grid)), ck, defines=rope, kernel_dir=KERNEL_DIR, bias=b,
                         extra=extra)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])
        ts, bf = MG.tile_bytes(cos.dtype), MG.tile_bytes(ttnn.bfloat16)
        entry["cbs"] += [MG._cb(7, core_grid, ts, cfg[0] * 2, cos.dtype), MG._cb(8, core_grid, ts, cfg[0] * 2, sin.dtype),
                         MG._cb(9, core_grid, bf, cfg[2], ttnn.bfloat16)]
        if qk_norm:   # c_4 carries [bias | g] a block; compute.cpp rope_block: y^2 (c_11), [q, k] partials (c_12), ones
            f32 = MG.tile_bytes(ttnn.float32)
            entry["cbs"] = [c for c in entry["cbs"] if c.format_descriptors[0].buffer_index != 4] + [
                MG._cb(4, core_grid, MG.tile_bytes(b.dtype), cfg[2] * 4, b.dtype),
                MG._cb(11, core_grid, bf, cfg[2], ttnn.bfloat16), MG._cb(12, core_grid, f32, 2 * cfg[0], ttnn.float32),
                MG._cb(13, core_grid, bf, 1, ttnn.bfloat16)]
        compute = entry["kernels"][4]
        compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
        compute.defines = [("FUSE_BIAS", "1")] + [(k, str(v)) for k, v in rope.items()]
        entry["pd"] = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["semaphores"], cbs=entry["cbs"])
        entry["extra"] = tuple(e.buffer_address() for e in extra)
        _CACHE[key] = entry
    addrs = (x.buffer_address(), w.buffer_address(), tuple(o.buffer_address() for o in outs), b.buffer_address())
    ea = tuple(e.buffer_address() for e in extra)
    if addrs != entry["addrs"] or ea != entry["extra"]:
        MG.rebind(entry, *addrs, extra_addrs=ea)
        entry["extra"] = ea
    try:
        ttnn.generic_op([x, w, *extra[2:], *outs], entry["pd"])
    except RuntimeError as exc:   # q / k placed in L1, but this program's circular buffers reach them: DRAM from now on
        if outs[0].memory_config().buffer_type != ttnn.BufferType.L1:
            raise
        from tt_bio import tenstorrent
        tenstorrent.absorb_l1_refusal("mm_rope qk L1", exc)
        _L1_REFUSED.add(tuple(shape))
        _CACHE.pop(key, None)
        for t in (*outs, *extra[2:]):
            ttnn.deallocate(t)
        return qkv_rope_heads(x, w, b, cos, sin, n_heads, cfg, grid, ckc, half, ncols, qk_norm=qk_norm)
    STATS[0] += 1
    return (*outs, *extra[2:])


def qk_norm_fold(w: torch.Tensor, b: torch.Tensor, gq: torch.Tensor, gk: torch.Tensor, d: int):
    """ESMC's q / k LayerNorm (over all d columns of q, no bias) split for the fused kernel: the centring folded into
    the [D, 3 d] qkv weight and [3 d] bias in float64 (each q / k output minus the mean over its d columns), and the
    [64, 3 d] bias tensor: row 0 the bias, rows 32-63 g (1 over v), so tile row 1 is g on every row for the epilogue's
    full-tile multiply. Pad / permute both as usual (`permute` keeps rows)."""
    w, b = w.double().clone(), b.double().clone()
    for s in (slice(0, d), slice(d, 2 * d)):
        w[:, s] -= w[:, s].mean(-1, keepdim=True)
        b[s] -= b[s].mean()
    g = torch.cat([gq.double().reshape(-1), gk.double().reshape(-1), torch.ones(d, dtype=torch.float64)])
    bg = torch.zeros(64, 3 * d, dtype=torch.float64)
    bg[0], bg[32:] = b, g
    return w.float(), bg.float()


def qk_rstd(stats, d: int, eps: float = 1e-5):
    """[rows, 64] bf16: columns 0-31 the rows' q rstd, 32-63 their k rstd, from `qkv_rope_heads(..., qk_norm=True)`'s
    partials (even tile columns q, odd k, each column of a tile the same row sum)."""
    nb = int(stats.padded_shape[-1]) // 64
    key = ("qk_rstd", nb, d, stats.device().id())
    sel = _CACHE.get(key)
    if sel is None:   # sums block j's q tile into columns 0-31 and its k tile into 32-63, / 32 for the repeat, / d
        s = torch.zeros(nb, 2, 32, 2, 32)
        for j in range(2):
            s[:, j, :, j, :] = 1.0 / (32 * d)
        sel = _CACHE[key] = ttnn.from_torch(s.reshape(64 * nb, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                            device=stats.device())
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    var = ttnn.matmul(stats, sel, compute_kernel_config=ck)
    r = ttnn.rsqrt(ttnn.add(var, eps))
    ttnn.deallocate(var)
    out = ttnn.typecast(r, ttnn.bfloat16)   # shsdpa's qk_rstd multiplies bf16 q / k tiles by it
    ttnn.deallocate(r)
    return out


# Heads narrower than 32 (SaProt 35M: head_dim 24). Two heads share a pair of column tiles, A = [h.x1 0.. | h+1.x1 0..]
# and B = [h.x2 0.. | h+1.x2 0..] (x1 / x2 the NeoX halves, each zero-padded to 16 columns), so rotate_half is the tile
# swap the epilogue already does for a 64-wide head. The writer then splits each rotated tile by faces: faces 0 / 2 are
# columns 0-15, faces 1 / 3 columns 16-31, so head h is [A[:, :16] | B[:, :16]] and head h + 1 is [A[:, 16:] | B[:, 16:]],
# each a [L, 32] head [x1 0.. | x2 0..]. q . k is unchanged by the dim order q and k share; v keeps `_pad_heads` order.

def half_pairs(w: torch.Tensor, b: torch.Tensor, n_heads: int, d: int, pad_heads=None):
    """torch Linear qkv `w` [3 H d, D] and `b` [3 H d] as the [D, 3 Hp 32] weight and [1, 3 Hp 32] bias of the half-pair
    layout: q and k as Hp / 2 tile pairs, v as Hp zero-padded 32-column heads. `pad_heads` (Hp >= H, even) adds zero
    heads so the tiles split over more in1 cores (`pad_heads_for`); the writer drops them (MM_HEADS_H)."""
    hp = pad_heads or n_heads
    assert n_heads % 2 == 0 and hp % 2 == 0 and hp >= n_heads and d % 2 == 0 and d // 2 <= 16, (n_heads, hp, d)
    h = d // 2
    ws, bs = w.reshape(3, n_heads, d, -1), b.reshape(3, n_heads, d)
    wo = w.new_zeros(3, hp, 32, w.shape[-1])
    bo = b.new_zeros(3, hp, 32)
    for j in range(2):          # q, k: pair p's tile A is virtual head 2p, tile B is 2p + 1
        for hd in range(n_heads):
            p, s = divmod(hd, 2)
            for half in range(2):
                wo[j, 2 * p + half, 16 * s:16 * s + h] = ws[j, hd, half * h:(half + 1) * h]
                bo[j, 2 * p + half, 16 * s:16 * s + h] = bs[j, hd, half * h:(half + 1) * h]
    wo[2, :n_heads, :d], bo[2, :n_heads, :d] = ws[2], bs[2]
    return wo.reshape(3 * hp * 32, -1).t().contiguous(), bo.reshape(1, -1)


def pad_heads_for(n_heads: int, ncols: int, n_block: int) -> int:
    """The fewest even heads >= n_heads whose half-pair qkv (3 Hp tiles) `layout` splits over `ncols` in1 cores."""
    hp = n_heads + n_heads % 2
    while layout(3 * hp, ncols, n_block) is None:
        hp += 2
    return hp


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
