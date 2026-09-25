"""Call census AND in-situ region timing for the ref-atom leg, in EVERY process.

`tt_bio.main predict` runs the fold in a `mp.get_context("spawn")` child, so a patch applied in
the parent never reaches the code that folds. `usercustomize` is imported by `site` in every
interpreter, including that child, which is why this lives here rather than in a driver.

It answers two questions per run:

  * did this process CALL the ref-atom leg, and which branch of `TT_BIO_OF3_DEVICE_REFATOM` did
    the call take -- the census, which is what says WHICH MODELS reach the path;
  * how long did each call take, at `perf_counter` resolution, inside the shipped inference
    process. tt_bio's own per-target runtime is rounded to 0.1 s and the whole effect of this
    flag is ~2e-2 s, so the fold line cannot resolve it and the region must be timed directly.

Counts and times are flushed after every call, so a run that is killed still leaves its census.
"""
import importlib.util
import json
import os
import sys
import threading
import time

TARGET = "tt_bio.openfold3_host_prep"
WRAP = ("ref_atom_embed", "ref_atom_embed_device", "run_input_atom_encoder")
OUT = os.environ.get("OF3T_CENSUS_OUT")

_lock = threading.Lock()
_census = {"pid": os.getpid(), "argv": sys.argv, "module_imported": False,
           "env_TT_BIO_OF3_DEVICE_REFATOM": os.environ.get("TT_BIO_OF3_DEVICE_REFATOM"),
           "calls": {}, "gate_branch": {}, "times_s": {}}


def _flush():
    if not OUT:
        return
    with _lock:
        path = f"{OUT}.{os.getpid()}.json"
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(_census, fh, indent=1)
        os.replace(tmp, path)


def _wrap(module):
    _census["module_imported"] = True
    for name in WRAP:
        fn = getattr(module, name, None)
        if fn is None or getattr(fn, "_of3t_censused", False):
            continue

        def make(fn=fn, name=name):
            def wrapper(*a, **kw):
                _census["calls"][name] = _census["calls"].get(name, 0) + 1
                if name == "run_input_atom_encoder":
                    # the gate is read inside the call, so read it the same way here
                    v = os.environ.get("TT_BIO_OF3_DEVICE_REFATOM")
                    b = "device_head" if v not in (None, "", "0", "false", "False") else "host_head"
                    _census["gate_branch"][b] = _census["gate_branch"].get(b, 0) + 1
                t0 = time.perf_counter()
                try:
                    return fn(*a, **kw)
                finally:
                    dt = time.perf_counter() - t0
                    _census["times_s"].setdefault(name, []).append(dt)
                    _flush()
            wrapper._of3t_censused = True
            wrapper.__name__ = name
            return wrapper

        setattr(module, name, make())
    _flush()


class _Proxy:
    def __init__(self, real):
        self._real = real

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        _wrap(module)

    def __getattr__(self, k):
        return getattr(self._real, k)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            spec = None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Proxy(spec.loader)
        return spec


if OUT:
    sys.meta_path.insert(0, _Finder())
    _flush()
