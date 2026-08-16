"""Apply the (K3, lever C) corner and prove it took effect -- in EVERY process.

`tt-bio predict` folds in a *spawned* child process, so a monkeypatch installed in the launcher
never reaches the code that resolves the constants: the first attempt at this produced a fold
with an empty probe file, which under the gate rules is an unscoreable run. Spawn re-execs
python with the parent environment, so a `sitecustomize` on PYTHONPATH is the one hook that
lands in the child as well as the parent.

Two env vars drive it:
  WHB2_PROBE       directory to write <pid>.json into. Required, else this file does nothing.
  WHB2_FORCE_SLMC  lever C: force SEQ_LEN_MORE_CHUNKING to this value after the arch tuning
                   has run. The shipped TT_BIO_SEQ_LEN_MORE_CHUNKING hook sits inside
                   _apply_grid_thresholds after its early return for full-size grids, so it is
                   unreachable on Blackhole; forcing after the call is the one mechanism that
                   works on both architectures.

K3 is TT_BIO_SDPA_DIV_K, read at import time, so it needs no help -- but it is reported here
too, because an arm is attributed by reading this probe and never by reading a command line.
"""
import json
import os
import sys
from importlib.machinery import PathFinder

# Chain-load the system sitecustomize this file shadows, so nothing it does is lost.
_sys_sc = "/usr/lib/python3.10/sitecustomize.py"
if os.path.exists(_sys_sc):
    try:
        import importlib.util as _u
        _s = _u.spec_from_file_location("_system_sitecustomize", _sys_sc)
        _m = _u.module_from_spec(_s)
        _s.loader.exec_module(_m)
    except Exception:
        pass

_PROBE = os.environ.get("WHB2_PROBE")
_FORCE = os.environ.get("WHB2_FORCE_SLMC")


def _install(T):
    orig = T._apply_grid_thresholds

    def probed(grid, device=None):
        orig(grid, device)
        if _FORCE:
            T.SEQ_LEN_MORE_CHUNKING = int(_FORCE)
        try:
            rec = {
                "pid": os.getpid(),
                "module": T.__file__,
                "grid": [int(grid[0]), int(grid[1])],
                "is_small_grid": bool(T._IS_SMALL_GRID),
                "forced_slmc": int(_FORCE) if _FORCE else None,
                "SEQ_LEN_MORE_CHUNKING": int(T.SEQ_LEN_MORE_CHUNKING),
                "SDPA_DIV_K": bool(T._SDPA_DIV_K),
                "k_chunk_640": int(T._dividing_sdpa_chunk_size(640)),
                "k_chunk_768": int(T._dividing_sdpa_chunk_size(768)),
                "fast_mode": bool(getattr(T, "_FAST_MODE", False)),
                "env": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith(("TT_BIO_", "TT_VISIBLE_", "WHB2_"))},
            }
            with open(os.path.join(_PROBE, f"{os.getpid()}.json"), "w") as f:
                json.dump(rec, f, indent=1)
        except Exception as e:  # a probe must never break a fold; a missing probe voids the run
            with open(os.path.join(_PROBE, f"{os.getpid()}.err"), "w") as f:
                f.write(repr(e))

    T._apply_grid_thresholds = probed


class _Hook:
    """Patch tt_bio.tenstorrent the moment it finishes executing, in whatever process."""

    TARGET = "tt_bio.tenstorrent"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = PathFinder.find_spec(fullname, path, target)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            _install(module)

        spec.loader.exec_module = exec_module
        return spec


if _PROBE:
    os.makedirs(_PROBE, exist_ok=True)
    sys.meta_path.insert(0, _Hook())
