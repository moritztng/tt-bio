"""of3t-bcastaudit call counter, loaded by every python process that has this directory on PYTHONPATH.

Active only when BCASTAUDIT_DIR is set. Wraps ttnn's binary multiply/add/subtract (and aliases,
in-place forms) plus ttnn.matmul, and records every call whose tensor operands differ in dtype,
split by whether their shapes broadcast, and every transpose_a matmul, keyed by the first caller
frame inside tt_bio/. The Tensor operators (`a * b`) resolve ttnn.multiply at call time, so they
are counted too. Each process writes BCASTAUDIT_DIR/<pid>.json on exit (and on multiprocessing
worker exit, which skips atexit).
"""
import os

_DIR = os.environ.get("BCASTAUDIT_DIR")

if _DIR:
    import atexit
    import collections
    import json
    import sys
    import time

    import ttnn

    _calls = collections.Counter()  # op -> calls wrapped
    _hits = collections.Counter()   # (op, form, site, dtype_a, dtype_b, shape_a, shape_b) -> calls
    _last = [time.time()]

    def _site():
        f = sys._getframe(2)
        while f is not None:
            fn = f.f_code.co_filename
            if "/tt_bio/" in fn:
                return f"{fn.split('/tt_bio/', 1)[1]}:{f.f_lineno}"
            f = f.f_back
        return "<outside tt_bio>"

    def _dump():
        rows = [dict(zip(("op", "form", "site", "dtype_a", "dtype_b", "shape_a", "shape_b"), k),
                     calls=n) for k, n in sorted(_hits.items(), key=lambda kv: -kv[1])]
        with open(os.path.join(_DIR, f"{os.getpid()}.json"), "w") as fh:
            json.dump({"pid": os.getpid(), "argv": sys.argv, "calls": dict(_calls),
                       "hits": rows}, fh, indent=1)

    def _record(op, form, a, b):
        _hits[(op, form, _site(), str(a.dtype).split(".")[-1], str(b.dtype).split(".")[-1],
               str(list(a.shape)), str(list(b.shape)))] += 1
        if time.time() - _last[0] > 10:
            _last[0] = time.time()
            _dump()

    def _wrap_binary(name):
        fn = getattr(ttnn, name)

        def wrapped(a, b, *args, **kw):
            _calls[name] += 1
            if isinstance(a, ttnn.Tensor) and isinstance(b, ttnn.Tensor) and a.dtype != b.dtype:
                _record(name, "mixed_bcast" if list(a.shape) != list(b.shape) else "mixed_same",
                        a, b)
            return fn(a, b, *args, **kw)

        setattr(ttnn, name, wrapped)

    for _n in ("multiply", "mul", "multiply_", "mul_", "add", "add_", "subtract", "sub",
               "subtract_", "sub_"):
        if hasattr(ttnn, _n):
            _wrap_binary(_n)

    _mm = ttnn.matmul

    def _matmul(a, b, *args, **kw):
        _calls["matmul"] += 1
        if kw.get("transpose_a"):
            _record("matmul", "transpose_a_mixed" if a.dtype != b.dtype else "transpose_a_same",
                    a, b)
        return _mm(a, b, *args, **kw)

    ttnn.matmul = _matmul
    atexit.register(_dump)
    try:
        from multiprocessing import util as _mpu
        _mpu.Finalize(None, _dump, exitpriority=100)
    except Exception:
        pass
