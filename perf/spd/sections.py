"""Wall time of named Python calls inside a bench.py run, device-synchronised at both ends.

    python perf/spd/sections.py SPEC [SPEC ...] -- <bench.py arguments>

SPEC is module:Class.method or module:function, e.g. tt_bio.openfold3_fold:OF3Fold._confidence. The
target is wrapped the moment its module is imported, so bench.py keeps its own import order (its host
thread cap must be in place before torch loads). Every call appends one record to <out>/sections.jsonl:
name, seconds, start time and call depth. The synchronize at entry charges queued device work to the
caller, not to the section; the one at exit charges the section's own device work to it.
"""
import importlib.abc, importlib.util, json, runpy, sys, time
from pathlib import Path

i = sys.argv.index("--")
SPECS, ARGV = sys.argv[1:i], sys.argv[i + 1:]
OUT = Path(ARGV[ARGV.index("--out") + 1])
by_mod = {}
for s in SPECS:
    mod, attr = s.split(":")
    by_mod.setdefault(mod, []).append(attr)
depth = [0]


def sync():
    from tt_bio import tenstorrent as T
    import ttnn
    if getattr(T, "_device", None) is not None:
        ttnn.synchronize_device(T._device)


def wrap(name, fn):
    def inner(*a, **k):
        sync(); t0 = time.perf_counter(); depth[0] += 1
        try:
            return fn(*a, **k)
        finally:
            sync(); depth[0] -= 1
            with open(OUT / "sections.jsonl", "a") as f:
                f.write(json.dumps(dict(name=name, s=time.perf_counter() - t0, t=time.time(), depth=depth[0])) + "\n")
    return inner


class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in by_mod:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        load = spec.loader.exec_module

        def exec_module(m):
            load(m)
            for attr in by_mod[name]:
                owner, _, meth = attr.rpartition(".")
                obj = getattr(m, owner) if owner else m
                setattr(obj, meth, wrap(f"{name.rsplit('.', 1)[-1]}:{attr}", getattr(obj, meth)))
        spec.loader.exec_module = exec_module
        return spec


OUT.mkdir(parents=True, exist_ok=True)
sys.meta_path.insert(0, Hook())
sys.argv = [str(Path(__file__).with_name("bench.py"))] + ARGV
runpy.run_path(sys.argv[0], run_name="__main__")
