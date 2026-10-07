"""Who holds device DRAM: every live ttnn tensor on the card, attributed to the first owner that reaches it.

The allocator says how much is held and nothing about by whom. This walks Python's object graph
from named roots in a fixed order and charges each DRAM buffer to the first root that reaches it,
deduplicated by buffer address so a view is not counted twice. The order puts the legitimate
owners first (the trunks the pool holds, the live tapes, the mask caches), so a buffer that is
ALSO reachable from a cache is charged to its rightful owner and only what nothing legitimate
holds lands on a cache's line.

The walk follows `gc.get_referents`, not `vars()`, because `autograd.Tensor` uses `__slots__` and
a tape is a web of them. It descends into closures and bound methods but not into modules,
classes or a function's globals, which would reach everything.

What no root reaches is swept last from `gc.get_objects()` and labelled by the type of the object
that holds it, and what no Python object holds at all is the allocator's total minus the sum.
"""
from __future__ import annotations

import gc
import sys
import types

import ttnn

_SKIP = (type, types.ModuleType, str, bytes, int, float, bool, type(None))
_ELEM = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1,
         "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024}


def dram(device) -> dict:
    """The allocator's own DRAM figures for the card, in bytes."""
    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    banks = int(mv.num_banks)
    return {"held": int(mv.total_bytes_allocated_per_bank) * banks,
            "free": int(mv.total_bytes_free_per_bank) * banks,
            "largest_per_bank": int(mv.largest_contiguous_bytes_free_per_bank),
            "total": int(mv.total_bytes_per_bank) * banks}


def _on_dram(t) -> int:
    """Bytes of `t` if it is an allocated DRAM tensor on a device, else 0."""
    try:
        if t.storage_type() != ttnn.StorageType.DEVICE or not t.is_allocated():
            return 0
        if t.memory_config().buffer_type != ttnn.BufferType.DRAM:
            return 0
        vol = 1
        for d in t.padded_shape:
            vol *= int(d)
        return int(vol * _ELEM.get(str(t.dtype).split(".")[-1].upper(), 2))
    except Exception:
        return 0


def _children(o):
    if isinstance(o, types.FunctionType):
        return [c.cell_contents for c in (o.__closure__ or ()) if _filled(c)] + \
               list(o.__defaults__ or ()) + list((o.__kwdefaults__ or {}).values())
    if isinstance(o, types.MethodType):
        return [o.__self__, o.__func__]
    if isinstance(o, types.FrameType):
        return list(o.f_locals.values())
    return gc.get_referents(o)


def _filled(cell) -> bool:
    try:
        cell.cell_contents
        return True
    except ValueError:
        return False


def census(roots, limit: int = 3_000_000) -> dict:
    """`{"by_owner": {label: [buffers, bytes]}, "attributed": bytes}` over `roots`, an ordered
    list of `(label, object)`; then everything else on the card, by holder type."""
    seen: set[int] = set()
    claimed: dict[int, str] = {}
    sizes: dict[int, int] = {}

    def charge(t, label):
        size = _on_dram(t)
        if not size:
            return
        try:
            addr = int(t.buffer_address())
        except Exception:
            addr = id(t)
        if addr not in claimed:
            claimed[addr], sizes[addr] = label, size

    for label, root in roots:
        stack = [root]
        while stack and len(seen) < limit:
            o = stack.pop()
            if isinstance(o, ttnn.Tensor):
                charge(o, label)
                continue
            if isinstance(o, _SKIP) or id(o) in seen:
                continue
            seen.add(id(o))
            stack.extend(_children(o))
    for holder in gc.get_objects():
        for r in gc.get_referents(holder):
            if isinstance(r, ttnn.Tensor):
                charge(r, f"elsewhere: {type(holder).__module__}.{type(holder).__qualname__}")
    by_owner: dict[str, list] = {}
    for addr, label in claimed.items():
        row = by_owner.setdefault(label, [0, 0])
        row[0] += 1
        row[1] += sizes[addr]
    return {"by_owner": dict(sorted(by_owner.items(), key=lambda kv: -kv[1][1])),
            "attributed": sum(sizes.values()), "walked": len(seen)}


def module_roots(prefix: str = "tt_bio"):
    """`(module.global, value)` for every loaded `prefix` module's globals that can hold data."""
    out = []
    for name, mod in sorted(sys.modules.items()):
        if not (name == prefix or name.startswith(prefix + ".")) or mod is None:
            continue
        short = name[len(prefix) + 1:] or prefix
        for g, v in vars(mod).items():
            if isinstance(v, _SKIP + (types.FunctionType, types.BuiltinFunctionType)):
                continue
            out.append((f"{short}.{g}", v))
    return out
