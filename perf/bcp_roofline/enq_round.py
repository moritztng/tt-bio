#!/usr/bin/env python3
"""How much of each device seam is Python enqueue, on the shipped round, no profiler.

`perf/bcx_p10_duotraj/duo_round.py` unchanged, plus one wrapper per seam and one around each
blocking readback (`ttnn.to_torch`, `ttnn.synchronize_device`). Inside a seam the host either
enqueues (busy) or waits on the card (blocked in one of those two calls), so

    enqueue = seam wall - time blocked in readback

and the card stays ahead of Python only while enqueue < card time for that seam. The blocked time
includes to_torch's host-side conversion of the seam's outputs (a few MB), so `enqueue` is a
slight UNDERestimate. Writes `<out>/enqueue.json`: one record per seam call.
"""
import json
import pathlib
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "perf" / "bcx_p10_duotraj"))
import duo_round as D                                                   # noqa: E402

_tls = threading.local()
RECORDS = []


def _blocking(mod, name):
    real = getattr(mod, name)

    def w(*a, **kw):
        t0 = time.perf_counter()
        try:
            return real(*a, **kw)
        finally:
            if getattr(_tls, "seam", None) is not None:
                _tls.blocked += time.perf_counter() - t0
                _tls.calls += 1
    setattr(mod, name, w)


def _install():
    import ttnn
    _blocking(ttnn, "to_torch")
    _blocking(ttnn, "synchronize_device")
    from tt_bio import bindcraft2 as B
    for module, cls in (("evoformer", B.EvoformerOnDevice), ("extra_msa", B.ExtraMsaOnDevice),
                        ("template", B.TemplateOnDevice)):
        for name in ("_primal", "_taped", "_backward"):
            orig = getattr(cls, name)

            def make(orig, tag):
                def wrapper(self, *a, **kw):
                    _tls.seam, _tls.blocked, _tls.calls = tag, 0.0, 0
                    t0 = time.perf_counter()
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        wall = time.perf_counter() - t0
                        RECORDS.append({"seam": tag, "wall": wall, "blocked": _tls.blocked,
                                        "enqueue": wall - _tls.blocked, "readbacks": _tls.calls,
                                        "t": time.time()})
                        _tls.seam = None
                return wrapper
            setattr(cls, name, make(orig, f"{module}:{name.lstrip('_')}"))


_install()

if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1]
    try:
        D.main()
    finally:
        pathlib.Path(out, "enqueue.json").write_text(json.dumps(RECORDS, indent=1))
