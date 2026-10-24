"""Run a script and record where tt-bio catches a device refusal, one JSON line per refusal.

    TRACE_RAISES_OUT=throws.jsonl python perf/of3_throws/trace_raises.py perf/of3_4xpd/xmodel_ab.py ARGS...

A refusal is a RuntimeError from ttnn saying L1 or DRAM declined a plan (a static-CB clash, CBs beyond
max L1, an allocator out-of-memory). Each one costs a TT_THROW the fold then absorbs, and caught throws
correlate with OF3 / OpenBind hangs (state/spd/note-from-spd-galaxy-g103-hang-clusters.md). The line
holds the message and the tt_bio stack from the innermost frame out, so the static gate that should
have declined the plan can be read off it. Line events are switched off in every frame, so the cost is
one Python call per Python call, paid only while tracing.

With TRACE_L1FITS_OUT set it also logs every distinct `_l1_fits` decision (caller, bytes, reserve, answer)
beside what the L1 allocator reports (bank size and count), so a gate that admits a tensor the device then
refuses can be read off the numbers.

With HANG_DUMP_AT=<s> (and HANG_DUMP_OUT=<file>) a process still running after <s> seconds dumps every thread's
Python stack every <s> seconds and, if PYSPY names a py-spy binary, its native stack three times 30 s apart. A healthy
cell is long done by then, so the dumps are the hang's location. The process lets any tracer attach to it
(PR_SET_PTRACER), since py-spy runs as its child and Yama scope 1 only lets ancestors trace. HANG_TRIAGE=<shell cmd>
runs once after the native dumps while the process is still hung (tt-triage against the chip it holds: the dispatch
timeout hook's triage prints only system_info, the hung process has to be alive and its Inspector logs readable).
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


def _patch_l1_fits(path):
    import ttnn
    from tt_bio import tenstorrent as T
    out, seen, inner = open(path, "a", buffering=1), set(), T._l1_fits

    def fits(nbytes, headroom, reserve_per_core=0):
        ok = inner(nbytes, headroom, reserve_per_core)
        f = sys._getframe(1)
        key = (f.f_code.co_name, f.f_lineno, int(nbytes), float(headroom), int(reserve_per_core), ok)
        if key not in seen:
            seen.add(key)
            try:
                mv = ttnn.get_memory_view(T.get_device(), ttnn.BufferType.L1)
                view = {"bank": int(mv.total_bytes_per_bank), "banks": int(mv.num_banks),
                        "free_bank": int(mv.total_bytes_free_per_bank),
                        "lcf": int(mv.largest_contiguous_bytes_free_per_bank)}
            except Exception as exc:
                view = {"err": str(exc)[:100]}
            out.write(json.dumps({"at": f"{f.f_code.co_name}:{f.f_lineno}", "bytes": int(nbytes), "headroom": headroom,
                                  "reserve": int(reserve_per_core), "ok": ok,
                                  "unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
                                  "grid": list(T.COMPUTE_GRID_MAIN), **view}) + "\n")
        return ok
    T._l1_fits = fits


def _hang_watch(at, path):
    """Neither dumper needs the GIL: a hang inside a native ttnn call holds it, and a Python watcher thread then never
    wakes (hcap 10-23: 2 hangs, 0 bytes dumped). faulthandler's timer is a C thread; py-spy runs from a shell."""
    import ctypes
    import faulthandler
    import subprocess
    ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)  # PR_SET_PTRACER, PR_SET_PTRACER_ANY
    out = open(path, "a", buffering=1)
    faulthandler.dump_traceback_later(at, repeat=True, file=out)
    spy = os.environ.get("PYSPY")
    if spy:
        sh = (f'sleep {at}; for i in 0 1 2; do kill -0 {os.getpid()} || exit; echo "=== dump $i native"; '
              f'timeout 120 {spy} dump --native --pid {os.getpid()}; sleep 30; done')
        if os.environ.get("HANG_TRIAGE"):
            sh += f'; kill -0 {os.getpid()} && echo "=== triage" && ({os.environ["HANG_TRIAGE"]})'
        subprocess.Popen(["sh", "-c", sh], stdout=out, stderr=out, start_new_session=True)

if __name__ == "__main__":
    if os.environ.get("HANG_DUMP_AT"):
        _hang_watch(float(os.environ["HANG_DUMP_AT"]), os.environ.get("HANG_DUMP_OUT", "hang.txt"))
    if os.environ.get("TRACE_L1FITS_OUT"):
        _patch_l1_fits(os.environ["TRACE_L1FITS_OUT"])
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    sys.settrace(_global)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.settrace(None)
        _out.close()
