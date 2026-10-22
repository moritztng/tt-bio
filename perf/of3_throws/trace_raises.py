"""Run a script and record where tt-bio catches a device refusal, one JSON line per refusal.

    TRACE_RAISES_OUT=throws.jsonl python perf/of3_throws/trace_raises.py perf/of3_4xpd/xmodel_ab.py ARGS...

A refusal is a RuntimeError from ttnn saying L1 or DRAM declined a plan (a static-CB clash, CBs beyond
max L1, an allocator out-of-memory). Each one costs a TT_THROW the fold then absorbs, and caught throws
correlate with OF3 / OpenBind hangs (state/spd/note-from-spd-galaxy-g103-hang-clusters.md). The line
holds the message and the tt_bio stack from the innermost frame out, so the static gate that should
have declined the plan can be read off it. Line events are switched off in every frame, so the cost is
one Python call per Python call, paid only while tracing.
"""
import json
import os
import runpy
import sys
import time

_KEYS = ("circular buffers", "Out of Memory", "beyond max L1", "out of memory")
_seen: set = set()
_out = open(os.environ.get("TRACE_RAISES_OUT", "throws.jsonl"), "a", buffering=1)
_t0 = time.time()


def _local(frame, event, arg):
    if event == "exception":
        exc = arg[1]
        if isinstance(exc, RuntimeError) and id(exc) not in _seen:
            msg = str(exc)
            if any(k in msg for k in _KEYS):
                _seen.add(id(exc))
                stack, f = [], frame
                while f is not None:
                    fn = f.f_code.co_filename
                    if "tt_bio" in fn:
                        stack.append(f"{fn.rsplit('tt_bio/', 1)[-1]}:{f.f_lineno}:{f.f_code.co_name}")
                    f = f.f_back
                line = next((l for l in msg.splitlines() if "TT_THROW" in l or any(k in l for k in _KEYS)), msg[:300])
                _out.write(json.dumps({"t": round(time.time() - _t0, 3), "msg": line[:400], "stack": stack[:12]}) + "\n")
    return _local


def _global(frame, event, arg):
    if "tt_bio" not in frame.f_code.co_filename:
        return None
    frame.f_trace_lines = False
    return _local


if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    sys.settrace(_global)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.settrace(None)
        _out.close()
