"""Name the reference cycles that keep device tensors alive through BindCraft 2's backward.

Runs perf/bgx_size/rung.py with Python's automatic collector off. Every `autograd.backward` call
is wrapped: when it returns, a `DEBUG_SAVEALL` collection moves everything that only a cycle kept
alive into gc.garbage. The script counts the ttnn tensors there (and their bytes), then walks
referrers inside that garbage from a few of them to print what closes each cycle (function
qualnames, frame code and line, container types). Then it frees the garbage and carries on.

    TT_VISIBLE_DEVICES=<chip> python perf/spd/bc2_gccycle.py --out O --target hPDL1 --binder 80 \
        --rounds 1 --footprint      (any rung.py arguments; SPD_GCCYCLE_MAX caps the reports, default 6)
"""
import collections, gc, json, os, runpy, sys, types

import ttnn
import tt_bio.autograd as ag

LIMIT = int(os.environ.get("SPD_GCCYCLE_MAX", "6"))
OUT = os.environ.get("SPD_GCCYCLE_OUT", "gccycle.jsonl")
_seen = {"calls": 0, "reports": 0, "depth": 0}


def _nbytes(t):
    try:
        n = 1
        for d in t.shape:
            n *= int(d)
        return n * (4 if t.dtype == ttnn.float32 else 2)
    except Exception:
        return 0


def _label(o):
    if isinstance(o, types.FunctionType):
        c = o.__code__
        return f"function {o.__qualname__} {os.path.basename(c.co_filename)}:{c.co_firstlineno}"
    if isinstance(o, types.FrameType):
        return f"frame {o.f_code.co_name} {os.path.basename(o.f_code.co_filename)}:{o.f_lineno}"
    if isinstance(o, types.CellType):
        return "cell"
    if isinstance(o, types.MethodType):
        return f"method {getattr(o.__func__, '__qualname__', '?')}"
    if isinstance(o, dict):
        return f"dict({len(o)}) keys {list(o)[:6]}"
    if isinstance(o, (list, tuple)):
        return f"{type(o).__name__}({len(o)})"
    return type(o).__module__ + "." + type(o).__qualname__


def _chains(start, garbage_ids, depth=8, width=3):
    """Referrer paths from `start` upward, staying inside the cyclic garbage."""
    out, frontier = [], [[start]]
    for _ in range(depth):
        nxt = []
        for path in frontier:
            refs = [r for r in gc.get_referrers(path[-1]) if id(r) in garbage_ids and r is not path
                    and all(r is not p for p in path)]
            if not refs:
                out.append(path)
            for r in refs[:width]:
                nxt.append(path + [r])
        frontier = nxt[:12]
        if not frontier:
            break
    out.extend(frontier)
    return [[_label(o) for o in p] for p in out[:6]]


def _report(tag):
    gc.set_debug(gc.DEBUG_SAVEALL)
    gc.collect()
    gc.set_debug(0)
    garbage = list(gc.garbage)
    gc.garbage.clear()
    tensors = [o for o in garbage if isinstance(o, ttnn.Tensor)]
    row = {"tag": tag, "call": _seen["calls"], "garbage": len(garbage), "ttnn_tensors": len(tensors),
           "ttnn_bytes": sum(_nbytes(t) for t in tensors),
           "types": collections.Counter(_label(o) if isinstance(o, types.FunctionType) else type(o).__qualname__
                                        for o in garbage).most_common(15)}
    if tensors and _seen["reports"] < LIMIT:
        _seen["reports"] += 1
        ids = {id(o) for o in garbage}
        big = sorted(tensors, key=_nbytes, reverse=True)[:2]
        row["chains"] = [{"shape": [int(d) for d in t.shape], "paths": _chains(t, ids)} for t in big]
    del garbage, tensors
    gc.collect()
    print("GCCYCLE " + json.dumps(row), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")


_backward = ag.backward


def backward(*a, **k):
    _seen["depth"] += 1
    try:
        return _backward(*a, **k)
    finally:
        _seen["depth"] -= 1
        _seen["calls"] += 1
        _report(f"backward depth {_seen['depth']}")


ag.backward = backward
gc.disable()
sys.argv = ["perf/bgx_size/rung.py"] + sys.argv[1:]
runpy.run_path("perf/bgx_size/rung.py", run_name="__main__")
