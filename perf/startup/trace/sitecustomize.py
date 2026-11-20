"""Startup-time phase accounting for one Python process (perf/startup/run.sh). Active only when ST_OUT names a file.

Wraps, from outside the model code, the calls that make up a cold start, and charges every wall-clock interval of
the main thread to the innermost active one: module imports (torch, ttnn, tt_bio, jax), device open, checkpoint
check/download, weight reads from disk, host->device tensor creation (ttnn.from_torch/as_tensor/to_device), trace
capture (begin..end_trace_capture), featurisation, and the top-level phases: model load and each prediction call.
Time inside a phase that no inner wrap claims is split by a 20 ms stack sampler into "import" (an importlib frame on
the stack) and the phase's own work. Writes one JSON line per top-level phase plus a final summary at exit.
"""
import os

if os.environ.get("ST_OUT"):
    import atexit, collections, functools, importlib.abc, importlib.machinery, json, sys, threading, time

    _out = open(os.environ["ST_OUT"], "a", buffering=1)
    _main = threading.main_thread().ident
    _now = time.monotonic
    with open("/proc/self/stat") as s:
        _ticks = int(s.read().rsplit(")", 1)[1].split()[19])
    with open("/proc/uptime") as u:
        _age = float(u.read().split()[0]) - _ticks / os.sysconf("SC_CLK_TCK")
    _T0 = _now() - _age                       # process start on the monotonic clock
    _stack = []                               # innermost last: category names
    _last = [_now()]
    _acc = collections.defaultdict(float)     # (top, cat) -> seconds
    _samp = collections.Counter()             # (top, class) -> samples while no inner wrap is active
    _hot = collections.Counter()              # (top, frame) -> samples, residual only
    _reads = set()
    TOP = ("load", "job", "device_open")
    _jobs = [0]
    _tops = []                                # (label, t_enter, t_leave)

    def _w(**kw):
        _out.write(json.dumps(kw) + "\n")

    def _top():
        for c in _stack:
            if c.split("#")[0] in TOP:
                return c
        return "pre" if not _jobs[0] else "post"

    def _tick():
        t = _now()
        _acc[(_top(), _stack[-1].split("#")[0] if _stack else "-")] += t - _last[0]
        _last[0] = t

    def _push(cat):
        _tick()
        if cat == "job":
            _jobs[0] += 1
            cat = f"job#{_jobs[0]}"
        if cat.split("#")[0] in TOP and not any(c.split("#")[0] in TOP for c in _stack):
            _tops.append([cat, _now() - _T0, None])
        _stack.append(cat)

    def _pop(cat):
        _tick()
        for i in range(len(_stack) - 1, -1, -1):
            if _stack[i].split("#")[0] == cat:
                c = _stack.pop(i)
                if c.split("#")[0] in TOP and not any(x.split("#")[0] in TOP for x in _stack):
                    _tops[-1][2] = _now() - _T0
                    _w(ev="top", label=c, t0=_tops[-1][1], t1=_tops[-1][2])
                return

    def _timed(fn, cat, path=True):
        @functools.wraps(fn)
        def w(*a, **kw):
            if threading.get_ident() != _main:
                return fn(*a, **kw)
            if path and cat == "read" and a:
                p = a[0]
                p = getattr(p, "name", p)
                if isinstance(p, (str, os.PathLike)):
                    _reads.add(os.path.abspath(os.fspath(p)))
            _push(cat)
            try:
                return fn(*a, **kw)
            finally:
                _pop(cat)
        return w

    def _marker(fn, push=None, pop=None):
        @functools.wraps(fn)
        def w(*a, **kw):
            r = fn(*a, **kw)
            if threading.get_ident() == _main:
                (_push if push else _pop)(push or pop)
            return r
        return w

    WRAP = {
        "torch": {"load": "read"},
        "numpy": {"load": "read"},
        "safetensors.torch": {"load_file": "read", "load": "read"},
        "huggingface_hub.file_download": {"hf_hub_download": "ckpt"},
        "huggingface_hub._snapshot_download": {"snapshot_download": "ckpt"},
        "ttnn": {"from_torch": "upload", "as_tensor": "upload", "to_device": "upload"},
        "tt_bio.tenstorrent": {"get_device": "device_open"},
        "tt_bio.worker": {"_WorkerState.load_model": "load", "_WorkerState.predict_one": "job",
                          "_ensure_local_artifacts": "ckpt"},
        "tt_bio.main": {"prepare_features": "featurize", "download_all": "ckpt"},
        "tt_bio.esmc": {"load_esmc": "load", "embed_sequences": "job"},
        "tt_bio.rfd4.engine": {"get_engine": "load"},
        "tt_bio.rfd4": {"design": "job"},
    }
    IMPORT_TIMED = {"torch": "import_torch", "ttnn": "import_ttnn", "tt_bio": "import_tt_bio", "jax": "import_jax"}

    class _SafeOpen:  # safetensors.safe_open is a Rust class; time the open and every tensor read through a proxy
        _orig = None

        def __init__(self, path, *a, **kw):
            _reads.add(os.path.abspath(os.fspath(path)))
            self._o = _timed(_SafeOpen._orig, "read", False)(path, *a, **kw)
            self._g = _timed(self._o.get_tensor, "read", False)

        def __enter__(self):
            self._o.__enter__()
            return self

        def __exit__(self, *e):
            return self._o.__exit__(*e)

        def get_tensor(self, k):
            return self._g(k)

        def __getattr__(self, n):
            return getattr(self._o, n)

    def _patch(mod):
        name = mod.__name__
        if name == "safetensors" and hasattr(mod, "safe_open"):
            _SafeOpen._orig = mod.safe_open
            mod.safe_open = _SafeOpen
        if name == "ttnn":
            for f, kw in (("begin_trace_capture", dict(push="trace_capture")),
                          ("end_trace_capture", dict(pop="trace_capture"))):
                if hasattr(mod, f):
                    setattr(mod, f, _marker(getattr(mod, f), **kw))
        for path, cat in WRAP.get(name, {}).items():
            owner_name, _, attr = path.rpartition(".")
            owner = getattr(mod, owner_name) if owner_name else mod
            if hasattr(owner, attr):
                setattr(owner, attr, _timed(getattr(owner, attr), cat))

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner, cat):
            self.inner, self.cat = inner, cat

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, mod):
            on = self.cat and threading.get_ident() == _main
            if on:
                _push(self.cat)
            try:
                self.inner.exec_module(mod)
            finally:
                if on:
                    _pop(self.cat)
            _patch(mod)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in WRAP and name not in IMPORT_TIMED and name != "safetensors":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec and spec.loader:
                spec.loader = _Loader(spec.loader, IMPORT_TIMED.get(name))
            return spec

    sys.meta_path.insert(0, _Finder())

    def _sample(dt=0.02):
        while True:
            time.sleep(dt)
            fr = sys._current_frames().get(_main)
            if fr is None or (_stack and _stack[-1].split("#")[0] not in TOP):
                continue
            top, imp, hot = _top(), False, None
            f = fr
            while f is not None:
                fn = f.f_code.co_filename
                if "importlib._bootstrap" in fn:
                    imp = True
                if hot is None and ("/tt_bio/" in fn or "/bindcraft/" in fn):
                    hot = f"{f.f_globals.get('__name__', '?')}:{f.f_code.co_name}"
                f = f.f_back
            _samp[(top, "import" if imp else "own")] += 1
            if not imp:
                _hot[(top, hot or f"{fr.f_globals.get('__name__', '?')}:{fr.f_code.co_name}")] += 1

    threading.Thread(target=_sample, daemon=True).start()

    def _fin():
        _tick()
        _w(ev="summary", end=_now() - _T0, age_at_site=_age, tops=_tops,
           acc=[[t, c, round(s, 4)] for (t, c), s in sorted(_acc.items())],
           samples=[[t, c, n] for (t, c), n in sorted(_samp.items())],
           hot=[[t, h, n] for (t, h), n in _hot.most_common(60)], reads=sorted(_reads))

    _done = []

    def _fin_once():
        if not _done:
            _done.append(1)
            _fin()

    atexit.register(_fin_once)
    _real_exit = os._exit

    def _exit(code):  # bench.py leaves through os._exit, which skips atexit
        _fin_once()
        _out.flush()
        _real_exit(code)
    os._exit = _exit
