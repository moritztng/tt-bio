"""How often does the OPM's whole path refuse, call by call, when a refusal is NOT remembered?

Runs perf/spd/bench.py unchanged (same argv) with OuterProductMean.__call__ wrapped: before every call the
row-block cap `_OPM_DRAM_ROW_CAP` is cleared (with --sticky it is left alone, today's behaviour), the device is
synchronised around the call, and the call's wall time and its OPM_ROW_STATS delta (whole / blocked /
dram_narrowed) go to <out>/opm_cap.jsonl. A call with dram_narrowed > 0 tried whole, was refused and finished
blocked, so its ms is the price of one refusal.

    python perf/spd_msa/opm_capfree.py [--sticky] -- <bench.py args>
"""
import importlib.abc, importlib.util, json, runpy, sys, time
from pathlib import Path

argv = sys.argv[1:]
sticky = argv[:1] == ["--sticky"]
if sticky:
    argv = argv[1:]
if argv[:1] == ["--"]:
    argv = argv[1:]
out = Path(argv[argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "opm_cap.jsonl", "a")


def patch(T):
    import ttnn
    orig, k = T.OuterProductMean.__call__, [0]

    def call(self, *args, **kw):
        if not sticky:
            T._OPM_DRAM_ROW_CAP.clear()
        dev = T.get_device()
        ttnn.synchronize_device(dev)
        before = dict(T.OPM_ROW_STATS)
        t0 = time.perf_counter()
        r = orig(self, *args, **kw)
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3
        delta = {key: T.OPM_ROW_STATS[key] - before[key] for key in before}
        LOG.write(json.dumps({"k": k[0], "sticky": sticky, "ms": ms, "delta": delta,
                              "cap": {str(key): v for key, v in T._OPM_DRAM_ROW_CAP.items()},
                              "t_unix": time.time()}) + "\n")
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
