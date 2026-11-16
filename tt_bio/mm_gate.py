"""`silu(x @ W1) * (x @ W2)` as ONE minimal_matmul with the gate in its epilogue (kernels/mm_gate).

    today   h1 = x @ W1 ; h2 = x @ W2 ; silu(h1) * h2       3 ops, h1 and h2 written to DRAM and read back
    gated   x @ [W1_0 | W2_0 | W1_1 | W2_1 | ...]          1 op, the gate computed from the accumulator in dest

The weight is W1 and W2 interleaved in column groups of half an N block (`interleave`), so every N block the
matmul accumulates is one group of fc1 columns followed by the same group of fc2 columns. The compute kernel's
epilogue (in place of its copy to the output CB) packs silu(fc1) * fc2 at half the block's width and the writer
addresses a [M, N] output with every N index halved. Everything else is the wheel's minimal_matmul through
`mm_generic.build`, so K blocking, multicast and the program shape are today's.

Wormhole ESMC's LM FFN (`esmc.SwiGLUFFN`): at the 300M page shape the gate pass alone is ~0.31 ms a layer and
fc1 writes both halves (~39 MB) where the gated op writes one.
"""
from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from . import mm_generic as MG

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "mm_gate"
TILE = 32

STATS = [0, 0]          # served, declined
_CACHE: dict = {}


def interleave(w1: torch.Tensor, w2: torch.Tensor, group: int) -> torch.Tensor:
    """[K, 2N]: W1 and W2 alternating in column groups of `group` columns (half an N block)."""
    k, n = w1.shape
    assert w2.shape == w1.shape and n % group == 0, (w1.shape, w2.shape, group)
    return torch.stack([w1.reshape(k, n // group, group), w2.reshape(k, n // group, group)], 2).reshape(k, 2 * n)


def layout(rows: int, n2: int, nb: int, grid):
    """(transpose, N tiles a core, M tiles a core) for N block `nb`, or None where no layout serves it.
    Every core's N range must be whole N blocks, so no block straddles a [W1_j | W2_j] group: N is padded to that
    (`mm_generic.build`'s n_align; a core past the real columns computes padding the writer drops). Of the two ways
    to lay M and N over the grid, the one with less padded work a core; the factory's own (N over the grid's
    columns unless M > N) on a tie. On Wormhole's 8x9 every ESMC shape divides and this is the factory's layout.
    Blackhole's 11 or 13 columns divide none of them, which kept the gate off there until N could be padded."""
    nt, mt = n2 // TILE, rows // TILE
    if nb % 2 or nt % nb:
        return None
    best = None
    natural = rows > n2
    for transpose in (natural, not natural):
        n_cores, m_cores = (grid[1], grid[0]) if transpose else (grid[0], grid[1])
        lay = (transpose, -(-nt // (n_cores * nb)) * nb, -(-mt // m_cores))
        if best is None or lay[1] * lay[2] < best[1] * best[2]:
            best = lay
    return best


def plan(x, n2: int, cfg, grid):
    """None when the gated program covers x @ [K, n2] at block config `cfg` on `grid`, else why not."""
    if cfg[2] % 2:
        return "n_block"
    if layout(x.volume() // int(x.shape[-1]), n2, cfg[2], grid) is None:
        return "n_split"
    return None


def config(x, n2: int, kt: int, grid, bf16_acc: bool, m_whole: bool = False):
    """The block config (M, K, N, subblock_h, subblock_w) for x @ [K, n2], or None where no N block fits `plan`.
    perf/spd_gap_whlm/gate_cfg_sweep.py (Galaxy chip, 1000 MHz, 3808 rows): this rule is the sweep's best or within
    2 % of it on ESMC 300M/600M, normal and --fast. N is the block up to 12 tiles whose `layout` pads the least work
    a core (the largest on a tie), the subblock the widest divisor of it that fits a dest half, K 4 tiles under bf16
    accumulation where K divides.
    M is 8 tiles unless a core's share of the rows is smaller: the ESMC-6B's 544 rows (17 tiles over 8 grid rows)
    run best at M 2 and K 4 (--shapes 6b: 0.706 ms against 0.834 at M 4 and 0.956 at M 8).
    `m_whole`: a core's share of 9-12 row tiles is one M block of 12, not two of 8 whose second is mostly padding.
    Blackhole p300 chip (11x10, 1350 MHz, HiFi3), share 11-12: 300M 0.407 -> 0.346 ms a layer, --fast 0.433 -> 0.379,
    600M 0.503 -> 0.469, --fast 0.552 -> 0.508, rel_l2 unchanged; each the sweep's best or within 2 % of it."""
    dmax = 8 if bf16_acc else 4
    rows = x.volume() // int(x.shape[-1])
    fits = [(nb, lay) for nb in (12, 10, 8, 6, 4) if (lay := layout(rows, n2, nb, grid)) is not None]
    if not fits:
        return None
    nb, (_, _, per_core) = min(fits, key=lambda f: f[1][1] * f[1][2])
    mb = 12 if m_whole and 8 < per_core <= 12 else min(8, 1 << (per_core.bit_length() - 1))
    kb = 4 if kt % 4 == 0 and (bf16_acc or mb <= 2) else 2
    sw = max(w for w in range(1, dmax + 1) if nb % w == 0)
    return (mb, kb, nb, max(h for h in (1, 2, 4, 8) if h * sw <= dmax and mb % h == 0), sw)


def overlay_ok() -> bool:
    """The PACK-thread epilogue's silu is calculate_silu_f32 only under the silu_f32 metal overlay; without it the
    wheel's approximate silu would run, so the gated path declines."""
    import os
    return "silu_f32" in os.environ.get("TT_BIO_METAL_OVERLAY", "").split(",")


def _f32_bits(v: float) -> str:
    import struct
    return f"{struct.unpack('<I', struct.pack('<f', v))[0]:#x}u"


def ln_weights(gamma: torch.Tensor, beta: torch.Tensor, w: torch.Tensor):
    """(C diag(gamma) W, beta W replicated over 32 rows) in float64 for an in-matmul LayerNorm over W [K, N]: then
    LayerNorm(x) W = rstd(x) * (x @ W'') + beta W, C = I - 1/K the centring matrix. Cast them to the device dtype."""
    g, b, w = gamma.double(), beta.double(), w.double()
    wg = g[:, None] * w
    return wg - wg.mean(0, keepdim=True), (b @ w).expand(32, -1).contiguous()


def gated(x, w12, cfg, ckc, grid, out_dtype=ttnn.bfloat16, probe=0, pack_sfpu=False, ln=None):
    """silu(x @ W1) * (x @ W2) for the interleaved `w12`, [.., N] in DRAM, or None when `plan` declines.
    `cfg` is minimal_matmul's (M_block, K_block, N_block, subblock_h, subblock_w) in tiles; the interleave
    group must be N_block / 2 tiles. `probe` (benchmarks only, wrong output) drops the epilogue's silu (1) or silu
    and multiply (2), to time its parts. `pack_sfpu` runs the epilogue's silu and multiply on the PACK thread (the
    silu is calculate_silu_f32 there only under the silu_f32 metal overlay).
    `ln` = (stats, bias32): x is the raw residual and w12 the interleaved `ln_weights` W'' of the LayerNorm before this
    FFN; stats is the producer's `mm_resid.matmul_add(..., rowstats=True)` output and bias32 the interleaved beta W
    [32, N2]. The epilogue scales the accumulator by the row's rstd and adds beta W before the gate."""
    n2 = int(w12.shape[-1])
    lay = layout(x.volume() // int(x.shape[-1]), n2, cfg[2], grid)
    if lay is None:
        STATS[1] += 1
        return None
    dev = x.device()
    shape = [int(d) for d in x.shape][:-1] + [n2 // 2]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    ck = MG.ckc_args(ckc)
    stats, bias = ln if ln is not None else (None, None)
    key = (spec(x), spec(w12), spec(out), tuple(cfg), tuple(grid), tuple(str(c) for c in ck), probe, pack_sfpu,
           ln is not None and (spec(stats), spec(bias)))
    entry = _CACHE.get(key)
    if entry is None:
        defines = {"MM_GATED": 1}
        if ln is not None:
            k = int(x.shape[-1])
            defines.update(MM_LN_RSTD=1, MM_LN_NB=int(stats.padded_shape[-1]) // 64,
                           MM_LN_INVK=_f32_bits(1.0 / k), MM_LN_EPS=_f32_bits(1e-5))
        entry = MG.build(dev, x, w12, [out], (tuple(cfg), tuple(grid)), ck, defines=defines,
                         kernel_dir=KERNEL_DIR, transpose=lay[0], n_align=cfg[2], bias=bias,
                         extra=(stats,) if ln is not None else ())
        if ln is not None:   # c_15 one row tile's statistics, c_14 the M block's rstd (compute.cpp ln_rstd_block)
            core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                          ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])
            f32 = MG.tile_bytes(ttnn.float32)
            entry["cbs"] += [MG._cb(15, core_grid, f32, 2 * defines["MM_LN_NB"], ttnn.float32),
                             MG._cb(14, core_grid, f32, cfg[0], ttnn.float32)]
        compute = entry["kernels"][4]
        compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
        compute.defines = ([(k_, str(v)) for k_, v in defines.items()]
                           + ([("MM_GATE_PROBE", str(probe))] if probe else [])
                           + ([("MM_GATE_PACK_SFPU", "1")] if pack_sfpu else []))
        entry["pd"] = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["semaphores"],
                                             cbs=entry["cbs"])
        entry["extra"] = (stats.buffer_address(),) if ln is not None else ()
        _CACHE[key] = entry
    addrs = (x.buffer_address(), w12.buffer_address(), (out.buffer_address(),)) + (
        (bias.buffer_address(),) if ln is not None else ())
    extra = (stats.buffer_address(),) if ln is not None else ()
    if addrs != entry["addrs"] or extra != entry.get("extra", ()):
        MG.rebind(entry, *addrs, extra_addrs=extra)
        entry["extra"] = extra
    ttnn.generic_op([x, w12, *((stats, bias) if ln is not None else ()), out], entry["pd"])
    STATS[0] += 1
    return out
