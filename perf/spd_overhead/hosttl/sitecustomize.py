"""Host timeline of a fold, per ttnn call site: time inside each ttnn call (host dispatch, or
blocking on a full queue / a read) and the Python time before it since the previous ttnn call
returned (host work the device may idle through). On only when HOSTTL_OUT names a JSON path;
put this directory first on PYTHONPATH. A snapshot is written after every Protenix.fold, so
the last one is the warm fold's.

Read it next to a device census: a site whose census idle gap is large and whose `before_s`
is large is host work in front of it; large `in_s` with a short device op is dispatch cost."""
import os
import sys

OUT = os.environ.get("HOSTTL_OUT")
OPS = ("concat", "permute", "to_layout", "reshape", "linear", "matmul", "layer_norm", "typecast",
       "from_torch", "to_torch", "add", "add_", "multiply", "multiply_", "deallocate", "reallocate",
       "pad", "slice", "transpose", "synchronize_device", "softmax", "embedding")

if OUT:
    import importlib.abc
    import json
    import time
    from collections import defaultdict

    stats = defaultdict(lambda: [0, 0.0, 0.0, 0.0])   # calls, in_s, before_s, max_before_s
    last = [time.perf_counter()]
    depth = [0]

    def _site():
        f = sys._getframe(2)
        while f is not None and ("/ttnn/" in f.f_code.co_filename or f.f_code.co_filename.endswith("/ops.py")):
            f = f.f_back
        if f is None:
            return "?"
        return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}:{f.f_code.co_name}"

    def wrap(name, fn):
        def w(*a, **k):
            if depth[0]:
                return fn(*a, **k)
            t0 = time.perf_counter()
            depth[0] = 1
            try:
                return fn(*a, **k)
            finally:
                depth[0] = 0
                t1 = time.perf_counter()
                s = stats[(name, _site())]
                s[0] += 1; s[1] += t1 - t0; b = t0 - last[0]; s[2] += b; s[3] = max(s[3], b)
                last[0] = t1
        return w

    def dump(tag):
        rows = sorted(({"op": k[0], "site": k[1], "calls": v[0], "in_s": round(v[1], 4),
                        "before_s": round(v[2], 4), "max_before_s": round(v[3], 4)}
                       for k, v in stats.items()), key=lambda r: -(r["in_s"] + r["before_s"]))
        with open(OUT, "a") as fh:
            fh.write(json.dumps({"tag": tag, "t": time.time(), "rows": rows}) + "\n")
        stats.clear()

    def patch_ttnn(m):
        for n in OPS:
            if hasattr(m, n):
                setattr(m, n, wrap(n, getattr(m, n)))

    def patch_protenix(m):
        orig = m.Protenix.fold

        def fold(self, *a, **k):
            stats.clear(); last[0] = time.perf_counter()
            try:
                return orig(self, *a, **k)
            finally:
                dump("fold")
        m.Protenix.fold = fold

    HOOKS = {"ttnn": patch_ttnn, "tt_bio.protenix": patch_protenix}

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in HOOKS:
                return None
            sys.meta_path.remove(self)
            try:
                import importlib.util
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def run(module):
                exec_module(module)
                HOOKS[name](module)
            spec.loader.exec_module = run
            return spec

    sys.meta_path.insert(0, _Finder())
