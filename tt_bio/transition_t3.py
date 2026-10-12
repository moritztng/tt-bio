"""Transition T3: the pair Transition (layer_norm -> SwiGLU -> fc3 [+ x]) as one program.

The incumbent runs a 512-token pair transition as ~11 row blocks of six to seven programs each (slice,
layer_norm, fc1 + silu, fc2, multiply_, fc3, add), every one ~0.1 ms and none of them near a roof. T3 gives
each core a contiguous run of row tiles (32 rows x C channels, C/32 contiguous pages) and does the whole
transition of a row tile in L1 against weights resident on every core, so x is read once and the result
written once, to the same pages (in place into x with `add_to_input`). Kernels: kernels/transition_t3.

Numerics differ from the incumbent by rounding only: LayerNorm is the trimul kernels' fp32 two-pass prologue,
fc2 enters the gate multiply from fp32 DST instead of a bf16 tensor, fc3 accumulates all hidden tiles in one
fp32 pass and fc3's bf16 output is added to x as the incumbent's add_ does. Lever `transition_t3`, ungraded.
"""

from __future__ import annotations

import struct

import ttnn

from . import dest_guard
from . import trimul_tail as TT

KERNEL_DIR = TT.KERNEL_DIR.parent / "transition_t3"
TILE = 32
# (Kt, Ht): c 128, hidden 512 (OpenFold3 / Boltz-2 / Protenix-v1); c 256, hidden 1024 (Protenix-v2), whose
# 768 weight tiles fit L1 only in bfp8 (`transition_b8`): the L1 check declines its bf16 weights.
# Protenix-v2 normal (bf16 weights) runs c 256 as two programs over hidden halves, key (8, 16) each (`split`).
KEYS = {(4, 16), (8, 32), (8, 16)}
STATS = [0, 0]              # served, declined
WHY: dict = {}              # decline reason -> calls
REFUSED: set = set()        # program keys the device refused (static-CB clash with live L1 buffers)
_CACHE: dict = {}
F3_SPLIT = 1               # fc3 K chunks (kernel arg 3); partials go through an fp32 CB between chunks


def cb_plan(kt, ht, res, stage=0):
    """T3's circular buffers per core, ([(index, bf16 tiles)], [(index, fp32 tiles)], [(index, weight
    tiles)]); N-independent. The weight CBs take the weights' own format. `stage` 1 / 2: the first / second
    program over hidden halves (`transition`, split): 1 writes its fc3 partial in fp32, 2 reads it back."""
    w = [(0, 2 * ht * kt),       # [W1 | W2], resident
         (1, ht * kt)]           # W3, resident
    bf = [(2, kt),               # layer_norm(x)
          (3, 4),                # silu(fc1) of one group
          (4, ht),               # h = silu(fc1) * fc2
          (7, 2 * kt),           # x, double buffered
          (8, kt), (9, kt),      # gamma, beta
          (10, 1)]               # 1/K reduce scaler
    if res:
        bf += [(5, 2 * kt),      # x for the residual add
               (15, kt)]         # fc3 output before it
    f32 = [(11, 1), (12, 1), (13, 1), (14, 1), (16, kt), (17, kt)]   # eps, mean, var, rstd, xc, tmp
    (f32 if stage == 1 else bf).append((6, 2 * kt))                   # output, double buffered
    if F3_SPLIT > 1 or stage == 2:
        f32 += [(18, kt)]                                             # fc3 partial (K chunks, or pass 1's)
    return bf, f32, w


def cb_bytes(kt, ht, res, wdtype=ttnn.bfloat16, stage=0):
    bf, f32, w = cb_plan(kt, ht, res, stage)
    return (sum(t for _, t in bf) * TT.BF16_TILE + sum(t for _, t in f32) * TT.FP32_TILE
            + sum(t for _, t in w) * TT.MG.tile_bytes(wdtype))


def split(nu, ncores):
    """Row tiles per core: [(first, count)], contiguous, the remainder on the first cores."""
    return TT.t1_split(nu, ncores)


def _no(why):
    STATS[1] += 1
    WHY[why] = WHY.get(why, 0) + 1
    return why


def eligible(x, ln_w, ln_b, w12, w3, stage=0):
    """None when T3 serves this call, else the reason it does not (counted in WHY)."""
    from .tenstorrent import lever
    from . import ops
    if not lever("transition_t3"):
        return "off"
    if ops.taping():
        return _no("tape")
    for t in (x, w12, w3):
        if t is None or t.layout != ttnn.TILE_LAYOUT:
            return _no("dtype")
        if t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            return _no("memcfg")
    if x.dtype != ttnn.bfloat16 or w12.dtype != w3.dtype or w12.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
        return _no("dtype")
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
    if cb_bytes(*key, stage != 1, w12.dtype, stage) > 0.85 * _l1_bank_bytes():
        return _no("l1")
    return None


def _build(x, ln_w, ln_b, w12, w3, out, grid, ckc, res, eps, stage=0, part=None):
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
        compile_time_args=[kt, ht, int(res), (inv << 16) | inv, eps_bits, int(stage == 2)]
        + acc(x) + acc(w12) + acc(w3) + acc(ln_w) + acc(ln_b) + acc(part if part is not None else x),
        runtime_args=rd, common_runtime_args=[0] * 6, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt] + acc(out),
        runtime_args=wr, common_runtime_args=[0], config=ttnn.WriterConfigDescriptor())
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ht, int(res), F3_SPLIT, int(stage == 2)], runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=True, dst_full_sync_en=full))
    bf, f32, w = cb_plan(kt, ht, res, stage)
    cbs = ([TT._cb(i, core_grid, t) for i, t in bf]
           + [TT._cb_fmt(i, core_grid, t, ttnn.float32) for i, t in f32]
           + [TT._cb_fmt(i, core_grid, t, w12.dtype) for i, t in w])
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def transition(x, ln_w, ln_b, w12, w3, ckc, grid, add_to_input=False, eps=1e-5, stage=0, part=None):
    """fc3(silu(fc1(xn)) * fc2(xn)) [+ x], xn = layer_norm(x; ln_w, ln_b, eps), with w12 = [W1 | W2] as
    [K, 2H] and w3 as [H, K] (x @ W layout). `add_to_input` writes the sum into x and returns x; otherwise a
    new DRAM tensor of x's shape. None when T3 does not serve the call; nothing is allocated or written then.
    `ckc` is mm_generic.ckc_args of the caller's silu config (its approx mode selects silu_f32)."""
    fid, approx, _, full = ckc
    if dest_guard.exposed_args((fid, approx, True, full)):
        _no("dest_guard")
        return None
    if fid == ttnn.MathFidelity.HiFi4 and F3_SPLIT == 1:
        # At HiFi4 the one-run 16-tile fc3 accumulation put isolated elements exactly 2^k low (WH .113: 2 of 8.4M
        # at N=256, 97 of 33.5M at 512, the same elements on four chips; HiFi3 and HiFi2 clean, silu exact or not).
        # The trunk runs HiFi3 (trunk_hifi3).
        _no("hifi4")
        return None
    if eligible(x, ln_w, ln_b, w12, w3, stage) is not None:
        return None
    key = ("t3", str(x.padded_shape), str(w12.padded_shape), str(w12.dtype), tuple(grid), tuple(str(c) for c in ckc),
           bool(add_to_input), float(eps), F3_SPLIT, stage)
    if key in REFUSED:
        _no("refused")
        return None
    res = stage != 1
    in_place = add_to_input and res
    out = x if in_place else ttnn.allocate_tensor_on_device(
        x.shape, ttnn.float32 if stage == 1 else ttnn.bfloat16, ttnn.TILE_LAYOUT, x.device(),
        ttnn.DRAM_MEMORY_CONFIG)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, ln_w, ln_b, w12, w3, out, grid, ckc, res, eps, stage, part)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), w12.buffer_address(), w3.buffer_address(),
                                  ln_w.buffer_address(), ln_b.buffer_address(),
                                  (part if part is not None else x).buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    io = [w12, w3, ln_w, ln_b] + ([part] if part is not None else []) + ([x] if in_place else [x, out])   # the output last, as generic_op returns it
    try:
        ttnn.generic_op(io, pd)
    except RuntimeError as exc:
        msg = str(exc)
        if "clash with L1 buffers" not in msg and "beyond max L1 size" not in msg:
            raise
        REFUSED.add(key)
        if not in_place:
            ttnn.deallocate(out)
        _no("l1_refused")
        return None
    STATS[0] += 1
    return out


def transition_split(x, ln_w, ln_b, halves, ckc, grid, add_to_input=False, eps=1e-5):
    """`transition` as two programs over hidden halves, for weights too large to sit in L1 whole (c 256 bf16):
    halves = [(w12_a, w3_a), (w12_b, w3_b)], each [W1 | W2] / W3 restricted to one half of the hidden channels.
    Program 1 writes fc3 over half a as an fp32 partial; program 2 recomputes layer_norm(x), accumulates fc3 over
    half b on top of it in fp32 DST and adds x, so the result rounds as the one-program F3_SPLIT = 2 path does.
    x is written only by program 2. None when either program does not serve; nothing is written then."""
    (wa, w3a), (wb, w3b) = halves
    if eligible(x, ln_w, ln_b, wb, w3b, 2) is not None:
        return None
    part = transition(x, ln_w, ln_b, wa, w3a, ckc, grid, eps=eps, stage=1)
    if part is None:
        return None
    out = transition(x, ln_w, ln_b, wb, w3b, ckc, grid, add_to_input=add_to_input, eps=eps, stage=2, part=part)
    ttnn.deallocate(part)
    return out


def needs_split(w12):
    """Whether [W1 | W2] (and its W3) is too large to sit in L1 whole but each hidden half fits a key."""
    from .tenstorrent import _l1_bank_bytes
    kt, ht = int(w12.shape[0]) // TILE, int(w12.shape[1]) // 2 // TILE
    return (ht % 2 == 0 and (kt, ht // 2) in KEYS
            and cb_bytes(kt, ht, True, w12.dtype) > 0.85 * _l1_bank_bytes())
