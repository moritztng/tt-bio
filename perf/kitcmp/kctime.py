"""Synced per-fold timer that rides inside any kit process, installed as kctime.pth in the kit's venv.

KC_WRAP="pkg.module:Class.method" names the model's own predict step (the same step the published stock cells timed).
A daemon thread wraps it as soon as the class exists and re-wraps it if a kit lever later replaces it, so the timer
always sits outermost. Each call: torch.cuda.synchronize() on both sides, one json line to $KC_TIMES with the seconds,
the pid and the qualified name of the function it wrapped (the kit's or upstream's). Unset KC_WRAP: nothing happens.
"""
import os, sys, threading, time, json

_depth = threading.local(); _depth.n = 0


def _run(spec, out):
    mod, _, attr = spec.partition(":"); cls_name, meth = attr.rsplit(".", 1)
    while True:
        m = sys.modules.get(mod)
        cls = getattr(m, cls_name, None) if m is not None else None
        f = getattr(cls, meth, None) if cls is not None else None
        if f is not None and not getattr(f, "_kctime", False):
            def wrap(orig):
                name = f"{getattr(orig, '__module__', '?')}.{getattr(orig, '__qualname__', '?')}"
                def w(*a, **k):
                    if getattr(_depth, "n", 0):  # a kit wrapper on top of ours calls back into us: time the outermost call only
                        return orig(*a, **k)
                    import torch
                    torch.cuda.synchronize(); t0 = time.perf_counter(); _depth.n = 1
                    try:
                        r = orig(*a, **k)
                    finally:
                        _depth.n = 0
                    torch.cuda.synchronize(); dt = time.perf_counter() - t0
                    with open(out, "a") as fh:
                        fh.write(json.dumps({"s": round(dt, 4), "pid": os.getpid(), "fn": name, "t": time.time()}) + "\n")
                    return r
                w._kctime = True
                return w
            setattr(cls, meth, wrap(f))
        time.sleep(0.05)


if os.environ.get("KC_WRAP") and os.environ.get("KC_TIMES"):
    threading.Thread(target=_run, args=(os.environ["KC_WRAP"], os.environ["KC_TIMES"]), daemon=True).start()
