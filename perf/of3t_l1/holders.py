"""Which taped tensors are still holding L1 when a program fails to lay out.

A circular-buffer clash names an ADDRESS and nothing else, so the buffer that collided has to
be identified from the tape's side. This walks every live `autograd.Tensor` the collector can
reach and reports the ones whose value is in L1, with the address the allocator gave it, its
shape, and the two lifetime bits that decide whether `free` was allowed to touch it.

`evictable` is the interesting one. `free` returns immediately on a non-evictable tensor: it
neither releases the buffer nor moves it to DRAM, so an L1-resident activation marked that way
occupies its place for the rest of the tape. Two things set it -- the three ops whose backward
reads their own output, and `_tape`'s storage-sharing check, which fires on every in-place verb.
"""
from __future__ import annotations

import gc


def l1_holders(ag, clash_addr=None, limit=60):
    out = {"tensors": [], "totals": {}}
    seen = set()
    for obj in gc.get_objects():
        if type(obj).__name__ != "Tensor" or not isinstance(obj, ag.Tensor):
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        v = getattr(obj, "value", None)
        try:
            if v is None or not v.is_allocated():
                continue
            if v.memory_config().buffer_type != ag.ttnn.BufferType.L1:
                continue
            rec = {"shape": [int(d) for d in v.shape], "dtype": str(v.dtype),
                   "addr": int(v.buffer_address()),
                   "evictable": bool(obj.evictable), "pinned": bool(obj.pinned),
                   "requires_grad": bool(obj.requires_grad),
                   "has_node": obj.node is not None}
        except Exception:                                                    # noqa: BLE001
            continue
        if clash_addr is not None:
            rec["is_clash_addr"] = rec["addr"] == int(clash_addr)
        out["tensors"].append(rec)
    out["tensors"].sort(key=lambda r: (-r.get("is_clash_addr", False), r["addr"]))
    t = out["tensors"]
    out["totals"] = {
        "l1_taped_tensors": len(t),
        "not_evictable": sum(1 for r in t if not r["evictable"]),
        "distinct_addrs": len({r["addr"] for r in t}),
        "at_clash_addr": sum(1 for r in t if r.get("is_clash_addr")),
    }
    out["tensors"] = t[:limit]
    return out
