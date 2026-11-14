"""Where does BindCraft 2's backward keep DRAM, block by block?

The cyclic-garbage probe (`bc2_gccycle.py`) found no device tensor held by a reference cycle, so
the suspect is an ordinary refcount-held one. This measures that directly: it wraps
`autograd.checkpoint`'s per-segment recompute and records free DRAM before and after each one,
so a segment that gives back less than it took shows up as a monotonic slide across blocks.
Also counts live `ttnn.Tensor` objects (gc.get_objects) and the live `autograd.Tensor`s holding
an allocated value, which separates "the tape still refers to it" from "ttnn has not freed it".

    TT_VISIBLE_DEVICES=<chip> SPD_BLOCKMEM_OUT=O/blockmem.jsonl \
        python perf/spd/bc2_blockmem.py --params ... --target hIL2R --binder 64 --rounds 1 --footprint
"""
import gc, json, os, runpy, sys

import ttnn
import tt_bio.autograd as ag
from tt_bio import tenstorrent

OUT = os.environ.get("SPD_BLOCKMEM_OUT", "blockmem.jsonl")
DEEP = os.environ.get("SPD_BLOCKMEM_DEEP") == "1"   # count live objects too (slow: walks the heap)
_n = {"i": 0}


def _free():
    if tenstorrent._device is None:
        return None, None
    mv = ttnn.get_memory_view(tenstorrent._device, ttnn.BufferType.DRAM)
    banks = int(mv.num_banks)
    return int(mv.total_bytes_free_per_bank) * banks, int(mv.total_bytes_per_bank) * banks


def _live():
    if not DEEP:
        return {}
    tt = at = alloc = 0
    for o in gc.get_objects():
        if isinstance(o, ttnn.Tensor):
            tt += 1
        elif isinstance(o, ag.Tensor):
            at += 1
            try:
                if o.value is not None and o.value.is_allocated():
                    alloc += 1
            except Exception:                                 # noqa: BLE001
                pass
    return {"live_ttnn": tt, "live_ag": at, "live_ag_allocated": alloc}


def _emit(**kw):
    with open(OUT, "a") as f:
        f.write(json.dumps(kw) + "\n")
    print("BLOCKMEM " + json.dumps(kw), flush=True)


_checkpoint = ag.checkpoint


def checkpoint(fn, *inputs, **kw):
    """Same checkpoint, but each segment's recompute is bracketed by a DRAM reading."""
    out = _checkpoint(fn, *inputs, **kw)

    def timed(inner, what):
        def fn_(*a, **k):
            i = _n["i"] = _n["i"] + 1
            before, total = _free()
            try:
                return inner(*a, **k)
            finally:
                after, _ = _free()
                _emit(seg=i, what=what, free_before=before, free_after=after,
                      kept=(before - after) if (before is not None and after is not None) else None,
                      total=total, **_live())
        return fn_

    # A single-output segment recomputes in node.fn; a multi-output one (a real Evoformer block)
    # only deposits its seed there and recomputes once in node.group, which every output of the
    # segment shares. So wrap each distinct group exactly once and give the wrapper to all of them.
    nodes = [n for n in (getattr(t, "node", None) if isinstance(t, ag.Tensor) else None
                         for t in (out if isinstance(out, (tuple, list)) else [out])) if n is not None]
    wrapped = {}
    for node in nodes:
        if node.group is not None:
            key = id(node.group)
            wrapped.setdefault(key, timed(node.group, "group"))
        elif node.fn is not None:
            node.fn = timed(node.fn, "fn")
    for node in nodes:
        if node.group is not None:
            node.group = wrapped[id(node.group)]
    return out


ag.checkpoint = checkpoint
import tt_bio.bindcraft2 as bc2  # noqa: E402  (it binds autograd as a module attribute, so patch above is seen)
assert bc2 is not None
sys.argv = ["perf/bgx_size/rung.py"] + sys.argv[1:]
runpy.run_path("perf/bgx_size/rung.py", run_name="__main__")
