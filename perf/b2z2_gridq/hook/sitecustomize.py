"""Install the SDPA pick census in EVERY python process, including spawned device workers.

`tt_bio.main` fans a fold out to one worker process per device, so a hook installed in the CLI
process records nothing: the Boltz-2 census came back 0 calls on a fold that plainly ran 1000s of
them. A `sitecustomize` on PYTHONPATH is inherited by the children, and a meta-path hook defers
the patch until `tt_bio.tenstorrent` is actually imported, so no process pays an import it would
not otherwise do.

    TT_BIO_PICK_CENSUS_DIR=<dir> PYTHONPATH=<this dir>:<worktree> python3 -m tt_bio.main ...

One json per pid. Join them with `model_pick_census.py --join`.
"""
import atexit, importlib.abc, importlib.machinery, json, os, sys
from collections import Counter

_DIR = os.environ.get("TT_BIO_PICK_CENSUS_DIR")

if _DIR:
    CALLS = Counter()

    def _dump():
        if not CALLS:
            return
        rows = []
        for (q, k, w, cap, rule, c), n in CALLS.items():
            rows.append({"q_len": q, "k_len": k, "work": w, "calls": n, "cores": c,
                         "shipped_chunk": cap, "rule_chunk": rule, "moves": cap != rule})
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, f"picks_{os.getpid()}.json"), "w") as fh:
            json.dump({"pid": os.getpid(), "argv": sys.argv[:6], "sdpa_calls": rows}, fh, indent=1)

    def _patch(T):
        orig = T._sdpa_program_config_for_lengths

        def wrapped(q_len, k_len, work=0):
            cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
            cap = T._capped_sdpa_chunk_size(q_len)
            rule = T._grid_q_chunk(q_len, int(work), cap, cores) if work else cap
            CALLS[(int(q_len), int(k_len), int(work), cap, rule, cores)] += 1
            return orig(q_len, k_len, work)

        wrapped.cache_clear = getattr(orig, "cache_clear", lambda: None)
        T._sdpa_program_config_for_lengths = wrapped
        for name in ("tt_bio.esmc", "tt_bio.esmfold2", "tt_bio.saprot"):
            m = sys.modules.get(name)
            if m is not None and hasattr(m, "_sdpa_program_config_for_lengths"):
                m._sdpa_program_config_for_lengths = wrapped

    class _Hook(importlib.abc.Loader):
        # Hold the ORIGINAL loader, not the spec: `spec.loader = _Hook(spec)` below makes
        # `self._spec.loader` this same object, and every call recurses until the stack dies.
        def __init__(self, loader):
            self._loader = loader

        def create_module(self, spec):
            return self._loader.create_module(spec)

        def exec_module(self, module):
            self._loader.exec_module(module)
            try:
                _patch(module)
            except Exception:                                        # noqa: BLE001
                pass

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            # The three modules that bind the picker by name are patched from the definition
            # site's hook, so only `tenstorrent` needs intercepting -- but it must be intercepted
            # BEFORE they import from it, which is what a meta-path finder guarantees.
            if fullname != "tt_bio.tenstorrent":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            sys.meta_path.remove(self)
            spec.loader = _Hook(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
    atexit.register(_dump)
