"""Per-call-site core_grid arm switch for ttnn.linear / ttnn.matmul.

The wrapper is installed once and stays installed in EVERY arm, so the host cost of the
indirection is identical on both sides of an A/B. An arm is a set of source line numbers in
tt_bio/tenstorrent.py; a call whose caller line is in the set, and which does not already carry
core_grid or a non-None program_config, gets core_grid=CORE_GRID_MAIN. Everything else is
untouched. Nothing here edits production source.

A site that passes program_config is NOT eligible: ttnn takes the program config as the complete
parallelisation decision and core_grid alongside it is either ignored or rejected. Those sites are
counted in `refused` so the census says so rather than silently pricing them at zero.
"""
from __future__ import annotations
import sys
from collections import Counter, defaultdict

_ORIG: dict[str, object] = {}
_SRC = ""
ARM: frozenset[int] = frozenset()
RECORD = False
CENSUS: dict[tuple, Counter] = defaultdict(Counter)
INJECTED = Counter()
REFUSED = Counter()
SEEN_LINES = Counter()


def _shape(t):
    try:
        return tuple(int(x) for x in t.shape)
    except Exception:                                                           # noqa: BLE001
        return None


def _wrap(name, orig, core_grid_main):
    def call(a, b, *args, **kwargs):
        f = sys._getframe(1)
        line = f.f_lineno if f.f_code.co_filename == _SRC else -1
        if line >= 0:
            SEEN_LINES[line] += 1
        if line in ARM:
            if "core_grid" in kwargs or kwargs.get("program_config") is not None:
                REFUSED[line] += 1
            else:
                kwargs["core_grid"] = core_grid_main
                INJECTED[line] += 1
        if RECORD and line >= 0:
            sa, sb = _shape(a), _shape(b)
            key = (line, name, sa, sb, str(kwargs.get("dtype")),
                   kwargs.get("program_config") is not None, "core_grid" in kwargs)
            CENSUS[key]["calls"] += 1
        return orig(a, b, *args, **kwargs)
    call.__name__ = name
    return call


def install(ttnn, tenstorrent):
    """Install the wrapper on ttnn.linear and ttnn.matmul. Idempotent."""
    global _SRC
    if _ORIG:
        return
    _SRC = tenstorrent.__file__
    for name in ("linear", "matmul"):
        orig = getattr(ttnn, name)
        _ORIG[name] = orig
        setattr(ttnn, name, _wrap(name, orig, tenstorrent.CORE_GRID_MAIN))


def set_arm(lines):
    global ARM
    ARM = frozenset(int(x) for x in lines)
    INJECTED.clear()
    REFUSED.clear()


def record(on: bool):
    global RECORD
    RECORD = on


def census_rows():
    rows = []
    for (line, name, sa, sb, dtype, has_pc, has_cg), c in sorted(CENSUS.items()):
        rows.append(dict(line=line, op=name, a=list(sa) if sa else None,
                         b=list(sb) if sb else None, dtype=dtype,
                         program_config=has_pc, core_grid=has_cg, calls=c["calls"]))
    return rows
