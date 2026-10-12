"""Transition T3: the pair Transition (layer_norm -> SwiGLU -> fc3 [+ x]) as one program.

The incumbent runs a 512-token pair transition as ~11 row blocks of six to seven programs each (slice,
layer_norm, fc1 + silu, fc2, multiply_, fc3, add), every one ~0.1 ms and none of them near a roof. T3 gives
each core a contiguous run of row tiles (32 rows x C channels, C/32 contiguous pages) and does the whole
transition of a row tile in L1 against weights resident on every core, so x is read once and the result
written once, to the same pages (in place into x with `add_to_input`). Kernels: kernels/transition_t3.

Numerics differ from the incumbent by rounding only: LayerNorm is the trimul kernels' fp32 two-pass prologue,
fc2 enters the gate multiply from fp32 DST instead of a bf16 tensor, fc3 accumulates all hidden tiles in one
fp32 pass and the residual add happens in fp32 before the one bf16 pack. Lever `transition_t3`, ungraded.
"""

from __future__ import annotations

import struct

import ttnn

from . import dest_guard
from . import trimul_tail as TT

KERNEL_DIR = TT.KERNEL_DIR.parent / "transition_t3"
TILE = 32
KEYS = {(4, 16)}            # (Kt, Ht): c 128, hidden 512 (OpenFold3 / Boltz-2 / Protenix pair transitions)
STATS = [0, 0]              # served, declined
WHY: dict = {}              # decline reason -> calls
REFUSED: set = set()        # program keys the device refused (static-CB clash with live L1 buffers)
_CACHE: dict = {}
WORMHOLE = False            # bench-only: serve on Wormhole too (perf/spd_trimul/t3.py --wh); no model path sets it


def cb_plan(kt, ht, res):
    """T3's circular buffers per core, ([(index, bf16 tiles)], [(index, fp32 tiles)]); N-independent."""
    bf = [(0, 2 * ht * kt),      # [W1 | W2], resident
          (1, ht * kt),          # W3, resident
          (2, kt),               # layer_norm(x)
          (3, 4),                # silu(fc1) of one group
          (4, ht),               # h = silu(fc1) * fc2
          (6, 2 * kt),           # output, double buffered
          (7, 2 * kt),           # x, double buffered
          (8, kt), (9, kt),      # gamma, beta
          (10, 1)]               # 1/K reduce scaler
    if res:
        bf.append((5, 2 * kt))   # x for the residual add
    f32 = [(11, 1), (12, 1), (13, 1), (14, 1), (16, kt), (17, kt)]   # eps, mean, var, rstd, xc, tmp
    return bf, f32


def cb_bytes(kt, ht, res):
    bf, f32 = cb_plan(kt, ht, res)
    return sum(t for _, t in bf) * TT.BF16_TILE + sum(t for _, t in f32) * TT.FP32_TILE


def split(nu, ncores):
    """Row tiles per core: [(first, count)], contiguous, the remainder on the first cores."""
    return TT.t1_split(nu, ncores)


def _no(why):
    STATS[1] += 1
    WHY[why] = WHY.get(why, 0) + 1
    return why


def eligible(x, ln_w, ln_b, w12, w3):
    """None when T3 serves this call, else the reason it does not (counted in WHY)."""
    from .tenstorrent import lever, _IS_SMALL_GRID
    from . import ops
    if not lever("transition_t3"):
        return "off"
    if _IS_SMALL_GRID and not WORMHOLE:
        return _no("wormhole")
    if ops.taping():
        return _no("tape")
    for t in (x, w12, w3):
        if t is None or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            return _no("dtype")
        if t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            return _no("memcfg")
    shp = [int(d) for d in x.shape]
    if [int(d) for d in x.padded_shape] != shp or len(shp) < 2 or shp[-1] % TILE or shp[-2] % TILE:
        return _no("shape")
    K = shp[-1]
    H2 = int(w12.shape[-1])
    if ([int(d) for d in w12.shape] != [K, H2] or H2 % (2 * TILE)
            or [int(d) for d in w3.shape] != [H2 // 2, K]):
        return _no("weight_shape")
    key = (K // TILE, H2 // 2 // TILE)
    if key not in KEYS:
        return _no("key=" + "x".join(map(str, key)))
    if not (TT._vec_ok(ln_w, K) and TT._vec_ok(ln_b, K)):
        return _no("ln_params")
    from .tenstorrent import _l1_bank_bytes
    if cb_bytes(*key, True) > 0.85 * _l1_bank_bytes():
        return _no("l1")
    return None


def _build(x, ln_w, ln_b, w12, w3, out, grid, ckc, res, eps):
    shp = [int(d) for d in x.shape]
    kt, ht = shp[-1] // TILE, int(w3.shape[-2]) // TILE
    nu = 1
    for d in shp[:-1]:
        nu *= d
    nu //= TILE
    gx, gy = grid
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (u0, n) in enumerate(split(nu, gx * gy)):
        cx, cy = c % gx, c // gx
        rd[cx][cy] = [u0, n]
        wr[cx][cy] = [u0, n]
        cp[cx][cy] = [n]
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / shp[-1]))[0] >> 16
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ht, int(res), (inv << 16) | inv, eps_bits]
        + acc(x) + acc(w12) + acc(w3) + acc(ln_w) + acc(ln_b),
        runtime_args=rd, common_runtime_args=[0] * 5, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt] + acc(out),
        runtime_args=wr, common_runtime_args=[0], config=ttnn.WriterConfigDescriptor())
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ht, int(res)], runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=True, dst_full_sync_en=full))
    bf, f32 = cb_plan(kt, ht, res)
    cbs = ([TT._cb(i, core_grid, t) for i, t in bf]
           + [TT._cb_fmt(i, core_grid, t, ttnn.float32) for i, t in f32])
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def transition(x, ln_w, ln_b, w12, w3, ckc, grid, add_to_input=False, eps=1e-5):
    """fc3(silu(fc1(xn)) * fc2(xn)) [+ x], xn = layer_norm(x; ln_w, ln_b, eps), with w12 = [W1 | W2] as
    [K, 2H] and w3 as [H, K] (x @ W layout). `add_to_input` writes the sum into x and returns x; otherwise a
    new DRAM tensor of x's shape. None when T3 does not serve the call; nothing is allocated or written then.
    `ckc` is mm_generic.ckc_args of the caller's silu config (its approx mode selects silu_f32)."""
    fid, approx, _, full = ckc
    if dest_guard.exposed_args((fid, approx, True, full)):
        _no("dest_guard")
        return None
    if eligible(x, ln_w, ln_b, w12, w3) is not None:
        return None
    key = ("t3", str(x.padded_shape), str(w12.padded_shape), tuple(grid), tuple(str(c) for c in ckc),
           bool(add_to_input), float(eps))
    if key in REFUSED:
        _no("refused")
        return None
    out = x if add_to_input else ttnn.allocate_tensor_on_device(
        x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, x.device(), ttnn.DRAM_MEMORY_CONFIG)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, ln_w, ln_b, w12, w3, out, grid, ckc, add_to_input, eps)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), w12.buffer_address(), w3.buffer_address(),
                                  ln_w.buffer_address(), ln_b.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    io = [w12, w3, ln_w, ln_b] + ([x] if add_to_input else [x, out])   # the output last, as generic_op returns it
    try:
        ttnn.generic_op(io, pd)
    except RuntimeError as exc:
        msg = str(exc)
        if "clash with L1 buffers" not in msg and "beyond max L1 size" not in msg:
            raise
        REFUSED.add(key)
        if not add_to_input:
            ttnn.deallocate(out)
        _no("l1_refused")
        return None
    STATS[0] += 1
    return out
