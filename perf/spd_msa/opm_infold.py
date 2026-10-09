"""Time every OuterProductMean call inside real folds, alternating a switch per call: a paired in-fold A/B.

Runs perf/spd/bench.py unchanged (same argv) with OuterProductMean.__call__ wrapped: the device is synchronised
before and after each call and the call's wall time goes to <out>/opm_calls.jsonl with its index and arm. FLAG
names a module global of tt_bio.tenstorrent (default _OPM_KPAD); call k runs with it on when
(k % 4 + k // 4) % 2 == 0, so each of the four MSA blocks alternates across recycles and both arms see every
block equally often under the same clock. The folds' outputs mix the two arms: read times here, not digests.

With --ops every ttnn call made inside the OPM is synchronised and timed too (op, input shapes, ms, arm), so a
call-level difference can be pinned on the op that carries it.

    python perf/spd_msa/opm_infold.py [--flag _OPM_KPAD] [--ops] -- <bench.py args>
"""
import importlib.abc, importlib.util, json, runpy, sys, time
from pathlib import Path

argv = sys.argv[1:]
flag = "_OPM_KPAD"
if argv[:1] == ["--flag"]:
    flag, argv = argv[1], argv[2:]
ops = argv[:1] == ["--ops"]
if ops:
    argv = argv[1:]
if argv[:1] == ["--"]:
    argv = argv[1:]
out = Path(argv[argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "opm_calls.jsonl", "a")


TIMED = {"matmul", "linear", "concat", "permute", "to_layout", "reshape", "zeros", "multiply_", "add", "add_",
         "reallocate", "typecast", "layer_norm"}


class Timed:
    """`ttnn` with the ops in TIMED synchronised and timed, for the duration of one OPM call."""

    def __init__(self, ttnn, dev, rec):
        self._t, self._dev, self._rec = ttnn, dev, rec

    def __getattr__(self, name):
        f = getattr(self._t, name)
        if name not in TIMED:
            return f

        def g(*a, **kw):
            t0 = time.perf_counter()
            r = f(*a, **kw)
            self._t.synchronize_device(self._dev)
            shp = [list(x.shape) for x in a if hasattr(x, "shape")][:2]
            if name == "concat" and a and isinstance(a[0], list):
                shp = [list(x.shape) for x in a[0]][:3] + [len(a[0])]
            self._rec.append((name, shp, (time.perf_counter() - t0) * 1e3))
            return r
        return g


def patch(T):
    import ttnn
    orig, k = T.OuterProductMean.__call__, [0]

    def call(self, *args, **kw):
        on = (k[0] % 4 + k[0] // 4) % 2 == 0
        setattr(T, flag, on)
        dev = T.get_device()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        rec = []
        if ops:
            # The OPM plan gate reads `ttnn is _SHIPPED_TTNN`, so the wrapper must stand in for both or
            # every --ops call silently times ttnn's auto plan instead of the shipped one.
            T.ttnn = T._SHIPPED_TTNN = Timed(ttnn, dev, rec)
        try:
            r = orig(self, *args, **kw)
        finally:
            T.ttnn = T._SHIPPED_TTNN = ttnn
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3
        LOG.write(json.dumps({"k": k[0], "flag": flag, "on": on, "ms": ms, "ops": rec, "t_unix": time.time()}) + "\n")
        LOG.flush()
        k[0] += 1
        return r

    T.OuterProductMean.__call__ = call


class Hook(importlib.abc.MetaPathFinder):
    """Wrap tt_bio.tenstorrent the moment bench.py imports it, so the arm's environment is already set."""

    def find_spec(self, name, path, target=None):
        if name != "tt_bio.tenstorrent":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(m):
            run(m)
            patch(m)
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, Hook())
sys.argv = [str(Path(__file__).resolve().parents[1] / "spd" / "bench.py"), *argv]
runpy.run_path(sys.argv[0], run_name="__main__")
