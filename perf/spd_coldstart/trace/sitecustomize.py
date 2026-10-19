"""Phase trace of a `tt-bio predict` CLI call across its parent and spawned worker (perf/spd_coldstart/cli.sh).

Active only when CS_TRACE names a file. Every python process started with this directory on PYTHONPATH appends
one JSON line per event: its start (process age when site ran), the moment tt_bio.main finished importing, entry and
exit of the CLI's phase functions below, and exit. Wrapping happens right after each module executes, so later
`from x import f` bindings see the wrapped function.
"""
import os

if os.environ.get("CS_TRACE"):
    import atexit
    import functools
    import importlib.abc
    import importlib.machinery
    import json
    import sys
    import time

    _f = open(os.environ["CS_TRACE"], "a", buffering=1)

    def _rec(ev, **kw):
        _f.write(json.dumps(dict(t=time.time(), pid=os.getpid(), ev=ev, **kw)) + "\n")

    with open("/proc/self/stat") as s:
        _ticks = int(s.read().rsplit(")", 1)[1].split()[19])
    with open("/proc/uptime") as u:
        _rec("proc_start", age_s=float(u.read().split()[0]) - _ticks / os.sysconf("SC_CLK_TCK"), argv=sys.argv[:2])
    atexit.register(lambda: _rec("exit"))

    WRAP = {
        "tt_bio.main": ["_spawn_worker_processes", "_stop_worker_processes", "_dispatch_run",
                        "_dispatch_to_controller"],
        "tt_bio.tenstorrent": ["get_device", "_close_device_locked"],
        "tt_bio.worker": ["run_worker_loop", "_WorkerState.load_model", "_WorkerState.predict_one", "prefetch_msas"],
    }

    def _wrap(owner, name, label):
        fn = getattr(owner, name, None)
        if fn is None:
            return

        @functools.wraps(fn)  # keeps spawn's pickle-by-name of run_worker_loop working
        def w(*a, **kw):
            _rec("enter", fn=label)
            t = time.monotonic()
            try:
                return fn(*a, **kw)
            finally:
                _rec("leave", fn=label, s=time.monotonic() - t)
        setattr(owner, name, w)

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, mod):
            self.inner.exec_module(mod)
            for path in WRAP.get(mod.__name__, ()):
                owner, _, name = path.rpartition(".")
                _wrap(getattr(mod, owner) if owner else mod, name, f"{mod.__name__}.{path}")
            _rec("imported", mod=mod.__name__)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in WRAP:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec and spec.loader:
                spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
