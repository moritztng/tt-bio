"""Log the bytes every ttnn trace capture uses, in every process a run spawns.

On PYTHONPATH as `perf/mgx_trace_region/hook`, with TRACE_REGION_LOG=<jsonl>. After each
end_trace_capture it appends one line: pid, the capture's caller (first tt_bio frame), the
region per DRAM bank, and TRACE bytes allocated per bank before and after (the difference is
this capture). It also records each device open's region and DRAM bank size. Measurement only:
nothing here changes what the run computes.
"""
import os

_LOG = os.environ.get("TRACE_REGION_LOG")

if _LOG:
    import importlib.abc
    import json
    import sys
    import time
    import traceback

    def _write(rec):
        rec.update(pid=os.getpid(), t=round(time.time(), 2))
        with open(_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _caller():
        for fr in reversed(traceback.extract_stack()[:-2]):
            if "/tt_bio/" in fr.filename:
                return f"{fr.filename.split('/tt_bio/')[-1]}:{fr.lineno}"
        return "?"

    def _patch(ttnn):
        begin, end = ttnn.begin_trace_capture, ttnn.end_trace_capture

        def used(dev):
            return ttnn.get_memory_view(dev, ttnn.BufferType.TRACE).total_bytes_allocated_per_bank

        def _begin(dev, *a, **k):
            _patch.before[id(dev)] = used(dev)
            return begin(dev, *a, **k)

        def _end(dev, tid, *a, **k):
            r = end(dev, tid, *a, **k)
            tv = ttnn.get_memory_view(dev, ttnn.BufferType.TRACE)
            dv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
            _write({"ev": "capture", "caller": _caller(),
                    "region_per_bank": tv.total_bytes_per_bank, "banks": tv.num_banks,
                    "trace_used_before": _patch.before.pop(id(dev), None),
                    "trace_used_after": tv.total_bytes_allocated_per_bank,
                    "dram_bank": dv.total_bytes_per_bank,
                    "dram_used_per_bank": dv.total_bytes_allocated_per_bank})
            return r

        _patch.before = {}
        ttnn.begin_trace_capture, ttnn.end_trace_capture = _begin, _end

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "ttnn":
                return None
            sys.meta_path.remove(self)
            try:
                import importlib.util
                spec = importlib.util.find_spec(name)
            finally:
                pass
            loader = spec.loader
            orig = loader.exec_module

            def exec_module(module):
                orig(module)
                _patch(module)
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
