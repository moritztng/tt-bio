"""Keep every fp32-accumulating matmul on Wormhole at K block 1.

On Wormhole, a matmul that accumulates more than one K tile in an fp32 dest writes a wrong value when
the next tile's partial sum nearly cancels the one already in dest: the result comes back as about
-2, -4 or -8 instead of a number near zero. It is deterministic (the same pixels on every run and
every chip) and data-dependent, about 1e-8 to 1e-7 of the outputs of a random matmul at HiFi4 and
less at HiFi3, so a relative-error check over a whole tensor does not see it
(perf/spd_wherr/cancel.py measures it directly, perf/spd_wherr/site_probe.py per shape).

At K block 1 dest holds one K tile at a time and the packer adds the partial sums in L1
(packer_l1_acc), which is not affected. So under fp32 dest acc on Wormhole this module:

* `install()` wraps ttnn.linear, ttnn.matmul and ttnn.experimental.minimal_matmul: a program config
  with in0_block_w > 1 is cloned at 1, an unconfigured 2D-weight call gets a 2D multicast config at 1,
  a minimal_matmul config gets K_block_size 1, and packer_l1_acc is switched on (without it the partials
  are reloaded into dest between K blocks, the same add).
* `descriptor()` does the same for the generic_op launchers that drive the minimal_matmul compute
  kernel (`mm_generic`), and `exposed()` tells kernels that keep all of K in dest (the trimul tail)
  to step aside.

`TT_BIO_DEST_GUARD=0` turns it off, for A/B only. Blackhole and bf16-dest calls are untouched.
"""
import functools
import os

import ttnn

ENABLED = os.environ.get("TT_BIO_DEST_GUARD", "1") != "0"
_ON = [False]                 # set by install() on a Wormhole device
STATS = {"rewritten": 0, "kept": 0, "refused": 0}
_REFUSED: dict = {}           # (op, shapes, config) -> error; the original call runs instead

_PC_FIELDS = ("compute_with_storage_grid_size", "in0_block_w", "out_subblock_h", "out_subblock_w",
              "out_block_h", "out_block_w", "per_core_M", "per_core_N", "transpose_mcast",
              "fused_activation", "fuse_batch", "mcast_in0", "gather_in0", "hop_cores",
              "num_global_cb_receivers", "untilize_out")


def active() -> bool:
    return _ON[0]


def exposed(ckc) -> bool:
    """True when `ckc` accumulates in an fp32 dest on a guarded (Wormhole) device."""
    return _ON[0] and ckc is not None and bool(getattr(ckc, "fp32_dest_acc_en", False))


def exposed_args(ckc_args) -> bool:
    """`exposed` for `mm_generic.ckc_args`'s (fidelity, approx, fp32_dest_acc_en, dst_full_sync)."""
    return _ON[0] and bool(ckc_args[2])


def descriptor(blk, ckc_args):
    """A generic minimal_matmul block (M, K, N, subblock_h, subblock_w) at K 1 when exposed."""
    if blk is None or not exposed_args(ckc_args) or blk[1] == 1:
        return blk
    return (blk[0], 1) + tuple(blk[2:])


_L1ACC: dict = {}


def _l1acc(ckc):
    """`ckc` with packer_l1_acc on (the same object when it already is)."""
    if getattr(ckc, "packer_l1_acc", False):
        return ckc
    key = (str(ckc.math_fidelity), bool(ckc.math_approx_mode), bool(getattr(ckc, "dst_full_sync_en", False)))
    if key not in _L1ACC:
        _L1ACC[key] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ckc.math_fidelity, math_approx_mode=ckc.math_approx_mode, fp32_dest_acc_en=True,
            packer_l1_acc=True, dst_full_sync_en=bool(getattr(ckc, "dst_full_sync_en", False)))
    return _L1ACC[key]


def _auto2d(a, b):
    """ttnn's 2D multicast at in0_block_w 1 for an unconfigured call on a 2D weight, or None."""
    ash, bsh = [int(d) for d in a.shape], [int(d) for d in b.shape]
    if len(bsh) > 2 and any(d != 1 for d in bsh[:-2]):
        return None
    if a.is_sharded():
        return None
    g = a.device().compute_with_storage_grid_size()
    m = 1
    for d in ash[:-1]:
        m *= d
    mt, nt = -(-m // 32), -(-bsh[-1] // 32)
    pm, pn = -(-mt // g.y), -(-nt // g.x)
    sw = max(s for s in range(1, min(4, pn) + 1) if pn % s == 0)
    sh = max(h for h in range(1, max(1, 4 // sw) + 1) if pm % h == 0)
    bh = max(h for h in range(sh, pm + 1, sh) if pm % h == 0 and (h * pn <= 32 or h == sh))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), in0_block_w=1, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=bh, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
        fused_activation=None, fuse_batch=True)


def _rewrite(op, a, b, kw):
    """The kwargs at K block 1, or None to run the call as it is."""
    kw2 = dict(kw)
    kw2["compute_kernel_config"] = _l1acc(kw["compute_kernel_config"])
    if op == "minimal_matmul":
        c = kw.get("config")
        if c is None:
            g = a.device().compute_with_storage_grid_size()
            kw2["config"] = ttnn.MinimalMatmulConfig(
                M_block_size=8, K_block_size=1, N_block_size=8, subblock_h=2, subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y))
        elif c.K_block_size != 1:
            kw2["config"] = ttnn.MinimalMatmulConfig(
                M_block_size=c.M_block_size, K_block_size=1, N_block_size=c.N_block_size,
                subblock_h=c.subblock_h, subblock_w=c.subblock_w,
                compute_with_storage_grid_size=c.compute_with_storage_grid_size)
        return kw2
    pc = kw.get("program_config")
    if pc is not None:
        if getattr(pc, "in0_block_w", 1) > 1:
            kw2["program_config"] = type(pc)(**{f: getattr(pc, f) for f in _PC_FIELDS if hasattr(pc, f)}
                                             | {"in0_block_w": 1})
        return kw2
    pc = _auto2d(a, b)
    if pc is None:
        return None
    kw2["program_config"] = pc
    kw2.pop("core_grid", None)
    return kw2


def _wrap(op, fn, operands):
    @functools.wraps(fn)
    def guarded(*args, **kw):
        if not exposed(kw.get("compute_kernel_config")):
            return fn(*args, **kw)
        a, b = operands(args, kw)
        key = (op, tuple(a.shape), tuple(b.shape), repr(kw.get("program_config") or kw.get("config")))
        if key in _REFUSED:
            STATS["refused"] += 1
            return fn(*args, **kw)
        kw2 = _rewrite(op, a, b, kw)
        if kw2 is None:
            STATS["kept"] += 1
            return fn(*args, **kw)
        try:
            y = fn(*args, **kw2)
        except Exception as e:   # a config the device refuses: the call runs as written, once noted
            _REFUSED[key] = str(e)[:300]
            STATS["refused"] += 1
            return fn(*args, **kw)
        STATS["rewritten"] += 1
        return y
    guarded.__wrapped_by_dest_guard__ = True
    return guarded


def install(device) -> bool:
    """Guard this process's matmuls if `device` is a Wormhole. Idempotent."""
    if not ENABLED or device.arch() != ttnn.Arch.WORMHOLE_B0:
        return False
    if not _ON[0]:
        mm = lambda args, kw: (args[0] if args else kw["input_tensor_a"],
                               args[1] if len(args) > 1 else kw["input_tensor_b"])
        mmm = lambda args, kw: (args[0] if args else kw["input_tensor"],
                                args[1] if len(args) > 1 else kw["weight_tensor"])
        ttnn.linear = _wrap("linear", ttnn.linear, mm)
        ttnn.matmul = _wrap("matmul", ttnn.matmul, mm)
        ttnn.experimental.minimal_matmul = _wrap("minimal_matmul", ttnn.experimental.minimal_matmul, mmm)
        _ON[0] = True
    return True
