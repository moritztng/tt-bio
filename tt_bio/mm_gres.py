"""`resid + gate * (x @ w + bias)` as ONE minimal_matmul (kernels/mm_gate under MM_GRES): a DiT block's adaLN-zero gated
residual in the output projection's epilogue, in place of the projection and a separate addcmul.

Today the token DiT writes `x @ w` to DRAM, then `addcmul_at` reads it back beside the residual and the gate and
writes the sum: two programs and one [M, N] round trip per gated residual, two per block. At BoltzGen's 514 x 768
token shape the addcmul is ~28 us of launch-sized work, 48 of them a diffusion step on Wormhole.

The non-writing DM kernel reads the residual's output block into c_5 (mm_resid's reader) and the gate's into c_6. The
gate may be a column window of a wider tensor, `(t, off, width, row)` as `tenstorrent._ColWin` carries it, read in
place: the kernel addresses output tile (i, j) at page `page0 + i * wt + j` of `t`, so `dit_cond_steps`' stacked
conditioning is never cut. The compute kernel adds the bias (a zero row where the projection has none), multiplies by
the gate and adds the residual in dest on the FPU, and packs once.
"""
import torch
import ttnn

from tt_bio import mm_generic as MG
from tt_bio.mm_gate import KERNEL_DIR
from tt_bio.rne_add import _NUM_CBS

STATS = [0, 0]   # runs, declines
_CACHE: dict = {}
_ZEROS: dict = {}


def _dev_key(dev):
    try:
        if hasattr(dev, "get_device_ids"):
            return ("mesh", tuple(dev.get_device_ids()))
        if hasattr(dev, "id"):
            return ("dev", int(dev.id()))
    except Exception:
        pass
    return id(dev)


def _zero_row(dev, n: int):
    key = (_dev_key(dev), n)
    z = _ZEROS.get(key)
    if z is None or not z.is_allocated():
        z = _ZEROS[key] = ttnn.from_torch(torch.zeros(1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          device=dev)
    return z


def gate_window(gate, m_tiles: int):
    """(tensor, page0, width-in-tiles) of a gate given as a tensor or a `(t, off, width, row)` column window, or None
    where the window is not tile-aligned."""
    if isinstance(gate, tuple):
        t, off, width, row = gate
    else:
        t, off, width, row = gate, 0, int(gate.shape[-1]), None
    ps = list(t.padded_shape)
    if off % 32 or width % 32 or ps[-1] % 32 or t.layout != ttnn.TILE_LAYOUT:
        return None
    wt = ps[-1] // 32
    rows_t = 1
    for d in ps[:-1]:
        rows_t *= d
    rows_t //= 32
    # a stepped tensor stacks the steps' [M, W] blocks on its leading dim; the window must be exactly one block (a
    # sample-batched x against a gate shared across samples is a broadcast the kernel does not do)
    per = rows_t if row is None else rows_t // ps[0]
    if per != m_tiles:
        return None
    r0 = 0 if row is None else row * m_tiles
    return t, r0 * wt + off // 32, wt


def matmul_gate_add(x, w, b, gate, resid, cfg, grid, ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, out_dtype=None):
    """resid + gate * (x @ w + b), [.., N] at `out_dtype` (default the residual's), for block config `cfg` (M, K, N, subblock_h, subblock_w) on `grid`, or
    None where the residual's or the gate's tile grid is not the output's (the caller keeps the projection and the
    addcmul). `b` may be None; `gate` a tensor or a `(t, off, width, row)` window of one."""
    n = int(w.padded_shape[-1])
    m_tiles = 1
    for d in list(x.padded_shape)[:-1]:
        m_tiles *= d
    m_tiles //= 32
    win = gate_window(gate, m_tiles)
    if (win is None or list(resid.padded_shape)[:-1] != list(x.padded_shape)[:-1] or int(resid.padded_shape[-1]) != n
            or resid.layout != ttnn.TILE_LAYOUT or (gate[2] if isinstance(gate, tuple) else int(gate.shape[-1])) != n):
        STATS[1] += 1
        return None
    g, page0, wt = win
    dev = x.device()
    ck = MG.ckc_args(ckc)
    # fp32 accumulation into an fp32 stream with no bias: the epilogue runs on the SFPU at fp32 (compute.cpp,
    # MM_GRES_FP32); the FPU path would round the residual to tf32
    fp32 = bool(ck[2]) and resid.dtype == ttnn.float32 and b is None
    if b is None:
        b = _zero_row(dev, n)
    shape = [int(d) for d in x.shape][:-1] + [int(w.shape[-1])]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype or resid.dtype, ttnn.TILE_LAYOUT, dev,
                                         memory_config)
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    key = (spec(x), spec(w), spec(b), spec(resid), spec(g), wt, spec(out), tuple(cfg), tuple(grid),
           tuple(str(c) for c in ck), fp32)
    entry = _CACHE.get(key)
    defs = {"MM_RESID": 1, "MM_GRES": 1, "MM_GRES_GATE_WT": wt}
    if entry is None:
        entry = MG.build(dev, x, w, [out], (tuple(cfg), tuple(grid)), ck, defines=defs, kernel_dir=KERNEL_DIR,
                         bias=b, extra=(resid, g))
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])
        entry["cbs"].append(MG._cb(5, core_grid, MG.tile_bytes(resid.dtype), cfg[0] * cfg[2], resid.dtype))
        entry["cbs"].append(MG._cb(6, core_grid, MG.tile_bytes(g.dtype), cfg[0] * cfg[2], g.dtype))
        compute = entry["kernels"][4]
        compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
        compute.defines = [("FUSE_BIAS", "1"), ("MM_RESID", "1"), ("MM_GRES", "1")] + ([("MM_GRES_FP32", "1")] if fp32 else [])
        if fp32:
            cfg_c = ttnn.ComputeConfigDescriptor(math_fidelity=ck[0], math_approx_mode=ck[1], fp32_dest_acc_en=True,
                                                 dst_full_sync_en=ck[3])
            modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
            modes[3] = modes[5] = ttnn.UnpackToDestMode.UnpackToDestFp32   # x W (the intermediate) and the residual
            cfg_c.unpack_to_dest_mode = modes
            compute.config = cfg_c
        # the DM kernels read [.., resid, gate, page0, outs]: build placed the two accessors' addresses, page0 goes
        # between them and the outputs, where rebind's extra_addrs slice (three long from here on) rewrites it
        nc = entry["n_chunks"]
        for name in ("in0_sender", "in0_recv", "in1_sender", "in1_recv"):
            for _, a in entry["rt"][name]:
                a.insert(len(a) - nc, page0)
        for k, name in zip(entry["kernels"][:4], ("in0_sender", "in0_recv", "in1_sender", "in1_recv")):
            k.runtime_args = entry["rt"][name]
        entry["pd"] = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["semaphores"], cbs=entry["cbs"])
        entry["extra"] = (resid.buffer_address(), g.buffer_address(), page0)
        _CACHE[key] = entry
    addrs = (x.buffer_address(), w.buffer_address(), (out.buffer_address(),), b.buffer_address())
    extra = (resid.buffer_address(), g.buffer_address(), page0)
    if addrs != entry["addrs"] or extra != entry["extra"]:
        MG.rebind(entry, *addrs, extra_addrs=extra)
        entry["extra"] = extra
    ttnn.generic_op([x, w, resid, g, out], entry["pd"])
    STATS[0] += 1
    return out
