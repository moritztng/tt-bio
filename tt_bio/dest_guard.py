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

Off by default: guarding everything costs +22 % of a Wormhole normal c730 fold (272 -> 333 s warm, AICLK 1000)
for a fault that writes about ten wrong values per fold; 1HCL scores 0.900 with it on and off. `TT_BIO_DEST_GUARD=1`
turns it on; a comma list of classes (pc, auto, mmm, gen, tri, tail), or joined by +, guards only those
(tri: the triangle-attention projections, the generic launches gen does not cover).
Blackhole and bf16-dest calls are untouched (Blackhole does not have the fault).
"""
import functools
import os

import ttnn

_CLASSES = ("pc", "auto", "mmm", "gen", "tri", "tail")
_ENV = os.environ.get("TT_BIO_DEST_GUARD", "0")
ENABLED = _ENV != "0"
GUARDED = frozenset(_CLASSES if _ENV in ("0", "1") else _ENV.replace("+", ",").split(","))
_ON = [False]                 # set by install() on a Wormhole device
STATS = {"rewritten": 0, "kept": 0, "refused": 0}
_REFUSED: dict = {}           # (op, shapes, config) -> the last refusal's error
_CHOSEN: dict = {}            # call key -> the kwargs it swaps in (False: run as written)

_PC_FIELDS = ("compute_with_storage_grid_size", "in0_block_w", "out_subblock_h", "out_subblock_w",
              "out_block_h", "out_block_w", "per_core_M", "per_core_N", "transpose_mcast",
              "fused_activation", "fuse_batch", "mcast_in0", "gather_in0", "hop_cores",
              "num_global_cb_receivers", "untilize_out")


def active() -> bool:
    return _ON[0]


def exposed(ckc) -> bool:
    """True when `ckc` accumulates in an fp32 dest on a guarded (Wormhole) device."""
    return _ON[0] and ckc is not None and bool(getattr(ckc, "fp32_dest_acc_en", False))


def tail_exposed(ckc) -> bool:
    """`exposed`, for a kernel that keeps all of K in dest and steps aside (the trimul tail)."""
    return exposed(ckc) and "tail" in GUARDED


def exposed_args(ckc_args, cls="tail") -> bool:
    """`exposed` for `mm_generic.ckc_args`'s (fidelity, approx, fp32_dest_acc_en, dst_full_sync)."""
    return _ON[0] and cls in GUARDED and bool(ckc_args[2])


def descriptor(blk, ckc_args, cls="gen"):
    """A generic minimal_matmul block (M, K, N, subblock_h, subblock_w) at K 1 when exposed."""
    if blk is None or not exposed_args(ckc_args, cls) or blk[1] == 1:
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


# Output block caps (tiles) tried in order for an unconfigured call: the largest block keeps ttnn's own
# operand reuse, a smaller one is the fallback when L1 is already holding something.
_AUTO_CAPS = (128, 32, 8)


def _auto2d(a, b, cap):
    """2D multicast at in0_block_w 1 for an unconfigured call on a 2D weight, or None."""
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
    bh = max(h for h in range(sh, pm + 1, sh) if pm % h == 0 and (h * pn <= cap or h == sh))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), in0_block_w=1, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=bh, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
        fused_activation=None, fuse_batch=True)


def _rewrite(op, a, b, kw):
    """Candidate kwargs at K block 1, best first; empty to run the call as it is."""
    kw2 = dict(kw)
    kw2["compute_kernel_config"] = _l1acc(kw["compute_kernel_config"])
    if op == "minimal_matmul":
        if "mmm" not in GUARDED:
            return []
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
        return [kw2]
    pc = kw.get("program_config")
    if pc is not None:
        if "pc" not in GUARDED:
            return []
        if getattr(pc, "in0_block_w", 1) > 1:
            kw2["program_config"] = type(pc)(**{f: getattr(pc, f) for f in _PC_FIELDS if hasattr(pc, f)}
                                             | {"in0_block_w": 1})
        return [kw2]
    if "auto" not in GUARDED:
        return []
    kw2.pop("core_grid", None)
    out = []
    for cap in _AUTO_CAPS:
        pc = _auto2d(a, b, cap)
        if pc is None:
            return []
        if not out or pc.out_block_h != out[-1]["program_config"].out_block_h:
            out.append(dict(kw2, program_config=pc))
    return out


def _wrap(op, fn, operands):
    @functools.wraps(fn)
    def guarded(*args, **kw):
        ckc = kw.get("compute_kernel_config")
        if not exposed(ckc):
            return fn(*args, **kw)
        a, b = operands(args, kw)
        cfg = kw.get("program_config") or kw.get("config")
        key = (op, tuple(a.shape), tuple(b.shape), repr(cfg), str(ckc.math_fidelity),
               bool(getattr(ckc, "packer_l1_acc", False)), "core_grid" in kw)
        swap = _CHOSEN.get(key)
        if swap is not None:     # the rewrite this call shape took the first time
            if swap is False:
                return fn(*args, **kw)
            STATS["rewritten"] += 1
            return fn(*args, **_apply(kw, swap))
        cands = _rewrite(op, a, b, kw)
        for kw2 in cands:
            try:
                y = fn(*args, **kw2)
            except Exception as e:   # a config the device refuses (L1): try the next one down
                _REFUSED[key] = str(e)[:300]
                continue
            _CHOSEN[key] = {k: v for k, v in kw2.items() if kw.get(k) is not v} | (
                {"core_grid": None} if "core_grid" in kw and "core_grid" not in kw2 else {})
            STATS["rewritten"] += 1
            return y
        _CHOSEN[key] = False
        STATS["refused" if cands else "kept"] += 1
        return fn(*args, **kw)
    guarded.__wrapped_by_dest_guard__ = True
    return guarded


def _apply(kw, swap):
    kw2 = dict(kw, **swap)
    if "core_grid" in swap:
        del kw2["core_grid"]
    return kw2


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
