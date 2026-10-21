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


def plan(x, n2: int, cfg, grid):
    """None when the gated program covers x @ [K, n2] at block config `cfg` on `grid`, else why not."""
    N_block = cfg[2]
    if N_block % 2:
        return "n_block"
    rows = x.volume() // int(x.shape[-1])
    nt = n2 // TILE
    transpose = rows > n2
    n_cores = grid[1] if transpose else grid[0]
    if nt % n_cores or (nt // n_cores) % N_block:
        return "n_split"           # every core's N range must be whole N blocks, so no block straddles a group
    return None


def config(x, n2: int, kt: int, grid, bf16_acc: bool, kb6: bool = False):
    """The block config (M, K, N, subblock_h, subblock_w) for x @ [K, n2], or None where no N block fits `plan`.
    perf/spd_gap_whlm/gate_cfg_sweep.py (Galaxy chip, 1000 MHz, 3808 rows): this rule is the sweep's best or within
    2 % of it on ESMC 300M/600M, normal and --fast. N is the largest plan-valid block up to 12 tiles, the subblock
    the widest divisor of it that fits a dest half, K 4 tiles under bf16 accumulation where K divides.
    M is 8 tiles unless a core's share of the rows is smaller: the ESMC-6B's 544 rows (17 tiles over 8 grid rows)
    run best at M 2 and K 4 (--shapes 6b: 0.706 ms against 0.834 at M 4 and 0.956 at M 8).
    `kb6` takes K in blocks of 6 under fp32 accumulation where K divides: gate_cfg_sweep --kb 2,3,4,6 (Galaxy chip,
    1000 MHz, HiFi3) 600M 1.131 -> 1.055 ms, 300M 0.833 -> 0.828, rel_l2 vs float64 unchanged."""
    dmax = 8 if bf16_acc else 4
    rows = x.volume() // int(x.shape[-1])
    per_core = -(-(rows // TILE) // (grid[0] if rows > n2 else grid[1]))
    mb = min(8, 1 << (per_core.bit_length() - 1))
    kb = 4 if kt % 4 == 0 and (bf16_acc or mb <= 2) else 6 if kb6 and not bf16_acc and kt % 6 == 0 else 2
    for nb in (12, 10, 8, 6, 4):
        cfg = (mb, kb, nb, 1, 1)
        if plan(x, n2, cfg, grid) is None:
            sw = max(w for w in range(1, dmax + 1) if nb % w == 0)
            return (mb, kb, nb, max(h for h in (1, 2, 4, 8) if h * sw <= dmax and mb % h == 0), sw)
    return None


def overlay_ok() -> bool:
    """The PACK-thread epilogue's silu is calculate_silu_f32 only under the silu_f32 metal overlay; without it the
    wheel's approximate silu would run, so the gated path declines."""
    import os
    return "silu_f32" in os.environ.get("TT_BIO_METAL_OVERLAY", "").split(",")


def gated(x, w12, cfg, ckc, grid, out_dtype=ttnn.bfloat16, probe=0, pack_sfpu=False):
    """silu(x @ W1) * (x @ W2) for the interleaved `w12`, [.., N] in DRAM, or None when `plan` declines.
    `cfg` is minimal_matmul's (M_block, K_block, N_block, subblock_h, subblock_w) in tiles; the interleave
    group must be N_block / 2 tiles. `probe` (benchmarks only, wrong output) drops the epilogue's silu (1) or silu
    and multiply (2), to time its parts. `pack_sfpu` runs the epilogue's silu and multiply on the PACK thread (the
    silu is calculate_silu_f32 there only under the silu_f32 metal overlay)."""
    n2 = int(w12.shape[-1])
    why = plan(x, n2, cfg, grid)
    if why is not None:
        STATS[1] += 1
        return None
    dev = x.device()
    shape = [int(d) for d in x.shape][:-1] + [n2 // 2]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    ck = MG.ckc_args(ckc)
    key = (spec(x), spec(w12), spec(out), tuple(cfg), tuple(grid), tuple(str(c) for c in ck), probe, pack_sfpu)
    entry = _CACHE.get(key)
    if entry is None:
        entry = MG.build(dev, x, w12, [out], (tuple(cfg), tuple(grid)), ck, defines={"MM_GATED": 1},
                         kernel_dir=KERNEL_DIR)
        compute = entry["kernels"][4]
        compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
        compute.defines = ([("MM_GATED", "1")] + ([("MM_GATE_PROBE", str(probe))] if probe else [])
                           + ([("MM_GATE_PACK_SFPU", "1")] if pack_sfpu else []))
        entry["pd"] = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["semaphores"],
                                             cbs=entry["cbs"])
        _CACHE[key] = entry
    addrs = (x.buffer_address(), w12.buffer_address(), (out.buffer_address(),))
    if addrs != entry["addrs"]:
        MG.rebind(entry, *addrs)
    ttnn.generic_op([x, w12, out], entry["pd"])
    STATS[0] += 1
    return out
