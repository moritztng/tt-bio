"""Every live device buffer, attributed to the container that holds it. One GC pass.

`ttnn.Tensor` is neither GC-tracked nor weak-referenceable, so a handle that outlives a step
cannot be found by asking the allocator or by holding a weakref. What CAN be enumerated is
every tracked container, and a handle that survives a step is by definition held by one, so
scanning the referents of every tracked object reaches every persistent holder and names it.

Attribution is by BUFFER ADDRESS, and a released address is reused within the same run, so a
per-step address diff reads as thousands of arrivals and departures that are the allocator
recycling one slot. The number that means something is the per-holder TOTAL, which is what
`table` reports and `growth` differences.
"""
from __future__ import annotations

import gc

_WIDTH = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0,
          "UINT8": 1.0, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625}


def _bytes(t) -> int:
    try:
        n = 1
        for d in t.padded_shape:
            n *= int(d)
        return int(n * _WIDTH.get(str(t.dtype).rsplit(".", 1)[-1].upper(), 2.0))
    except Exception:
        return 0


def _registry(t):
    """Which of the engine's two handle registries holds this wrapper, if either."""
    from tt_bio import autograd as ag
    v = getattr(t, "_value", None)
    if v is None:
        return "detached"
    if ag._WRAPPED.get(id(v)) is t:
        return "in _WRAPPED"
    if ag._PARAMS.get(id(v)) is t:
        return "in _PARAMS"
    return "unregistered"


def _known():
    """The engine's own long-lived registries, by identity, so a dict can be named."""
    from tt_bio import autograd as ag
    out = {id(ag._WRAPPED): "autograd._WRAPPED", id(ag._PARAMS): "autograd._PARAMS",
           id(ag._ZERO_CACHE): "autograd._ZERO_CACHE", id(ag._CKPT_PINS): "autograd._CKPT_PINS"}
    try:
        from tt_bio import taped_ttnn as tt
        for name in dir(tt):
            v = getattr(tt, name, None)
            if isinstance(v, (dict, list, set)) and name.isupper():
                out[id(v)] = f"taped_ttnn.{name}"
    except Exception:
        pass
    return out


def _tag(holder, known) -> str:
    k = known.get(id(holder))
    if k is not None:
        return k
    t = type(holder)
    if t is dict:
        return "dict"
    if t in (list, tuple, set, frozenset):
        return t.__name__
    name = f"{t.__module__}.{t.__qualname__}"
    if name == "tt_bio.autograd.Tensor":
        return ("autograd.Tensor " + _registry(holder)
                + (" pinned" if getattr(holder, "pinned", False) else "")
                + (" grad" if getattr(holder, "requires_grad", False) else "")
                + (" taped" if getattr(holder, "node", None) is not None else ""))
    return name


def registries():
    from tt_bio import autograd as ag
    return {"_WRAPPED": len(ag._WRAPPED), "_PARAMS": len(ag._PARAMS),
            "_ZERO_CACHE": len(ag._ZERO_CACHE), "_CKPT_PINS": len(ag._CKPT_PINS),
            "_TOUCHED": len(ag._TOUCHED)}


def census(*, collect: bool = True):
    """{address: (bytes, tag)} over every allocated DRAM buffer a container holds."""
    import ttnn
    if collect:
        gc.collect()
    known = _known()
    out = {}
    for o in gc.get_objects():
        try:
            refs = gc.get_referents(o)
        except Exception:
            continue
        for r in refs:
            if type(r) is not ttnn.Tensor:
                continue
            try:
                if not r.is_allocated():
                    continue
                if str(r.memory_config().buffer_type).rsplit(".", 1)[-1] != "DRAM":
                    continue
                a = int(r.buffer_address())
            except Exception:
                continue
            if a not in out:
                out[a] = (_bytes(r), _tag(o, known))
    return out


def table(c):
    """{tag: (count, bytes)}."""
    by = {}
    for _a, (n, tag) in c.items():
        cnt, b = by.get(tag, (0, 0))
        by[tag] = (cnt + 1, b + n)
    return by


def total(c) -> int:
    return sum(v[0] for v in c.values())


def growth(before, after, *, top: int = 10) -> str:
    """What each holder gained since the last census. Totals, not arrivals."""
    b, a = table(before), table(after)
    rows = []
    for tag in set(b) | set(a):
        c0, n0 = b.get(tag, (0, 0))
        c1, n1 = a.get(tag, (0, 0))
        rows.append((n1 - n0, c1 - c0, c1, n1, tag))
    rows.sort(key=lambda r: -abs(r[0]))
    head = (f"held {total(before)/1e9:.3f} -> {total(after)/1e9:.3f} GB "
            f"in {len(before)} -> {len(after)} buffers")
    body = "".join(f"\n    {d/1e9:+8.3f} GB ({dc:+5d} -> {c:6d} buffers, {n/1e9:7.3f} GB)  {tag}"
                   for d, dc, c, n, tag in rows[:top])
    return head + body
