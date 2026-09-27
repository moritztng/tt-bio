#!/usr/bin/env python3
"""bcx-p10-mmroof leg 1: the round's arithmetic calls keyed by shape, placement and PLAN.

`bcx-p10-mmlay` widened `OpTimer`'s second key from the op name to shape + placement and used
it to find the calls that hand `ttnn.matmul` no plan at all. It served 696 a round and declined
2316 because they already carried a `program_config` -- and its key wrote those down as the
single token `pc`, so every call with a plan looked the same as every other call with a plan.

That token is what this row opens. A `MatmulMultiCoreReuseMultiCastProgramConfig` carries a
compute grid, `in0_block_w`, `out_subblock_h/w`, `out_block_h/w`, `per_core_M/N`,
`transpose_mcast` and `fuse_batch`; `MatmulMultiCoreReuseProgramConfig` carries a subset and
`...MultiCast1D...` a different one. Every field is a performance decision. So the key here
carries the FACTORY NAME and the field values, and two calls of the same shape running two
different plans land in two different rows of the table instead of one.

It also carries the compute kernel config -- math fidelity, `fp32_dest_acc_en`,
`packer_l1_acc` -- because `fp32_dest_acc_en` halves the destination register file to 4 tiles
and therefore bounds `out_subblock_h * out_subblock_w`, which is a plan constraint and not a
precision footnote.

Nothing else moves: the same `device = synced - free - lambda * calls` subtraction, the same
records, the same medians, the same FLOP counting off padded shapes. A_plan is comparable to
`bcx-p10-mmlay`'s A_shape and to `bcx-p10-calls`'s A_verb because it is the same measurement
re-keyed a third time.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_mmlay import mmkey as MK          # noqa: E402

#: Short names for the matmul program-config factories. The factory is the first thing the plan
#: says: it picks the program, and the fields only mean anything relative to it.
_FACTORY = {
    "MatmulMultiCoreReuseMultiCastProgramConfig": "RMC2D",
    "MatmulMultiCoreReuseMultiCast1DProgramConfig": "RMC1D",
    "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig": "RMCDS",
    "MatmulMultiCoreReuseProgramConfig": "REUSE",
    "MatmulMultiCoreProgramConfig": "MULTI",
    "MatmulMultiCoreNonOptimizedReuseProgramConfig": "NOOPT",
}

#: Read in this order so the digest is stable across factories that share a field. Anything the
#: object does not have is simply absent from the digest rather than guessed at.
_FIELDS = (
    ("compute_with_storage_grid_size", "g"),
    ("in0_block_w", "w"),
    ("out_subblock_h", "sh"),
    ("out_subblock_w", "sw"),
    ("out_block_h", "bh"),
    ("out_block_w", "bw"),
    ("per_core_M", "M"),
    ("per_core_N", "N"),
    ("fuse_batch", "fb"),
    ("transpose_mcast", "tm"),
    ("mcast_in0", "mc0"),
    ("gather_in0", "gi0"),
    ("fused_activation", "act"),
)

_FID = {"HiFi4": "hifi4", "HiFi3": "hifi3", "HiFi2": "hifi2", "LoFi": "lofi"}


def _scalar(v):
    """A grid size prints as `11x10`, a bool as 0/1, everything else as its repr tail."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    for ax in (("x", "y"), ("width", "height")):
        if all(hasattr(v, a) for a in ax):
            return "%dx%d" % (int(getattr(v, ax[0])), int(getattr(v, ax[1])))
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return "%dx%d" % (int(v[0]), int(v[1]))
    s = str(v).rsplit(".", 1)[-1]
    return s[:12]


def plan_digest(pc) -> str:
    """`RMC2D:g11x10,w9,sh1,sw1,bh26,bw26,M26,N26,fb0,tm0` for a live program config.

    `none` when the call carries no plan, which is what `bcx-p10-mmlay`'s lever serves.
    """
    if pc is None:
        return "none"
    name = type(pc).__name__
    head = _FACTORY.get(name, name[:10])
    bits = []
    for attr, short in _FIELDS:
        if not hasattr(pc, attr):
            continue
        try:
            v = getattr(pc, attr)
        except Exception:                                                     # noqa: BLE001
            continue
        if attr == "fused_activation" and v is None:
            continue
        bits.append("%s%s" % (short, _scalar(v)))
    return head + ":" + ",".join(bits)


def kernel_digest(ck) -> str:
    """`hifi4/f32acc/l1acc` for a live compute kernel config, `-` when the call has none.

    `fp32_dest_acc_en` is in the key because it halves the dest register file to 4 tiles and so
    caps `out_subblock_h * out_subblock_w`: it bounds the plan, it is not a precision footnote.
    """
    if ck is None:
        return "-"
    bits = []
    fid = getattr(ck, "math_fidelity", None)
    if fid is not None:
        s = str(fid).rsplit(".", 1)[-1]
        bits.append(_FID.get(s, s.lower()))
    for attr, tag in (("fp32_dest_acc_en", "f32acc"), ("packer_l1_acc", "l1acc"),
                      ("math_approx_mode", "approx"), ("dst_full_sync_en", "fullsync")):
        try:
            if bool(getattr(ck, attr)):
                bits.append(tag)
        except Exception:                                                     # noqa: BLE001
            pass
    return "/".join(bits) if bits else "-"


def _pc_of(kwargs):
    return kwargs.get("program_config") or kwargs.get("config")


def _ck_of(kwargs):
    return kwargs.get("compute_kernel_config")


class PlanOpTimer(MK.ShapeOpTimer):
    """`ShapeOpTimer` with the matmul family's key widened again, to the PLAN.

    The base class already emits `op#BxMxKxN#flags#a:..#b:..#o:..[#extra]`, where `extra`
    collapses a whole program config to the token `pc`. This appends `#plan:<digest>` and
    `#ck:<digest>` after it, so the key is a strict extension: every token `bcx-p10-mmlay`
    keyed on is still there and in the same place, and `mmkey.parse` still reads it.
    """

    def verb_key(self, path, args, kwargs, out):
        key = super().verb_key(path, args, kwargs, out)
        if "#" not in key:
            return key
        return "%s#plan:%s#ck:%s" % (key, plan_digest(_pc_of(kwargs)), kernel_digest(_ck_of(kwargs)))


def parse(key):
    """`mmkey.parse` plus the two plan tokens. `plan` is `none` for a call with no config."""
    d = MK.parse(key)
    d.setdefault("plan", "none")
    d.setdefault("ck", "-")
    if "#" in key:
        for p in key.split("#")[3:]:
            if p.startswith("plan:"):
                d["plan"] = p[5:]
            elif p.startswith("ck:"):
                d["ck"] = p[3:]
    d["factory"] = d["plan"].split(":", 1)[0]
    return d


def plan_fields(plan: str) -> dict:
    """`RMC2D:g11x10,w9,M26,N26` -> `{"factory": "RMC2D", "g": "11x10", "w": 9, ...}`."""
    out = {"factory": plan.split(":", 1)[0]}
    if ":" not in plan:
        return out
    for bit in plan.split(":", 1)[1].split(","):
        if not bit:
            continue
        i = 0
        while i < len(bit) and not (bit[i].isdigit() or bit[i] == "-"):
            i += 1
        k, v = bit[:i], bit[i:]
        out[k] = int(v) if v.isdigit() else v
    return out
