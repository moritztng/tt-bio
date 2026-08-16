"""D: half the output drain of the trimul in-projection issued on the other NOC.

The in-projection reads 128 MiB and writes 512 MiB per call. Both halves leave through the DM
kernel that `minimal_matmul` makes the output writer, on one NOC, while the other DM kernel's NOC
carries only the 128 MiB read. `noc_async_write_tile` takes an explicit NOC index, so every second
output tile can go out on the quiet wire without a second RISC, a second circular buffer or a
semaphore. Same tiles, same addresses, same bytes, same order: bit-exact by construction, and
`torch.equal` against the shipped call at the production shape.

MEASURED on qb2 card 0, `[512,512,256] x [256,1024]`, batched 8 calls per synchronize, median of
5 batches, 3 rounds (`perf/esmbeat/s_d_dualnoc.json`):

    native 3.154 ms   generic 3.125 (0.991x, torch.equal)
    nowrite 2.104     so the DRAM write is 1.021 ms, 33 % of the op
    dualnoc 2.884     -0.240 ms/call, 1.308x on the write half, torch.equal

A/A floor 0.069 ms and dualnoc reads 2.884 / 2.881 / 2.893 across the rounds, so the delta is 3.5
floors wide. 1084 in-projection calls per 512 aa fold.

The write is NOT NOC-saturated, which is why this returns a quarter of the write and not a half:
512 MiB in 1.021 ms would be 526 GB/s against a 429.7 GB/s measured DRAM roof, so most of the write
already overlaps compute and only its non-overlapped tail is being paid.

`DM_DYNAMIC_NOC` is not a tuning choice. Under the default `DM_DEDICATED_NOC` the firmware only
runs `noc_local_state_init` for the kernel's own NOC, so a write issued on the other one never
issues and `noc_async_write_barrier` spins forever. That is a first-call device hang at 100 % host
CPU that looks exactly like a wedged card and is not one.

The gates below are deliberately narrow: bf16 both sides, interleaved DRAM both sides, an
unpadded activation, a 2-D weight, and the M > N orientation the transcription was verified under.
Anything else falls through to the stock op.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from . import mm_generic as G

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "mm_split"
TILE = 32

# (calls served, calls that fell through to the stock op)
STATS = [0, 0]
# Why calls were refused, keyed by (reason, shape). A gate that never fires has to say why.
REJECTS: dict = {}

TRIMUL_IN_PROJ_DUAL_NOC = False
_ENABLED = os.environ.get(
    "TT_BIO_TRIMUL_DUAL_NOC", "1" if TRIMUL_IN_PROJ_DUAL_NOC else "0") == "1"


def set_enabled(on: bool) -> None:
    global _ENABLED
    _ENABLED = bool(on)


def enabled() -> bool:
    return _ENABLED


def _reject(reason, shape):
    REJECTS[(reason, tuple(shape))] = REJECTS.get((reason, tuple(shape)), 0) + 1
    STATS[1] += 1
    return None


def in_proj(x, w, ckc, dtype, memory_config):
    """`minimal_matmul(x, w)` with half the drain on the other NOC, or `None` to leave it alone.

    Byte-identical to the unconfigured `ttnn.experimental.minimal_matmul` the caller would
    otherwise run: the block config passed in is `_MM_DEFAULT`, which IS what
    `determine_default_block_sizes` returns under `fp32_dest_acc_en`.
    """
    if not _ENABLED:
        return None
    shape = [int(d) for d in x.shape]
    if dtype != ttnn.bfloat16 or x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16:
        return _reject("dtype", shape)
    if x.layout != ttnn.TILE_LAYOUT or len(w.shape) != 2:
        return _reject("layout_or_rank", shape)
    if [int(d) for d in x.padded_shape] != shape:
        # A padded activation changes which tile ids the writer addresses; the transcription was
        # verified unpadded and the F1 tail already shipped one non-tile-aligned crash.
        return _reject("padded_activation", shape)
    xmc, wmc = x.memory_config(), w.memory_config()
    if (memory_config is None or memory_config.buffer_type != ttnn.BufferType.DRAM
            or xmc.buffer_type != ttnn.BufferType.DRAM
            or xmc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
            or wmc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
        return _reject("memory", shape)

    from .tenstorrent import _MM_DEFAULT, COMPUTE_GRID_MAIN
    n = int(w.shape[-1])
    m = 1
    for d in shape[:-1]:
        m *= d
    if m <= n:
        # transpose_core_grid flips below that, a core-grid orientation this has never run on
        return _reject("m_le_n", shape)
    if m % TILE or n % TILE or int(w.shape[-2]) % TILE:
        return _reject("not_tile_aligned", shape)

    dev = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape[:-1] + [n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, memory_config)
    G.generic_minimal_matmul(
        dev, x, w, out, (_MM_DEFAULT, tuple(COMPUTE_GRID_MAIN)), G.ckc_args(ckc),
        {"MM_DUAL_NOC": 1}, KERNEL_DIR, None, ttnn.NOC_MODE.DM_DYNAMIC_NOC)
    STATS[0] += 1
    return out
