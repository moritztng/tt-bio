"""Make the pinned PXDesign source importable against the protenix the box has.

PXDesign pins a protenix from the 0.5 era; qb1's reference venv carries protenix 2.0, which
moved four data modules under `protenix.data.core`, moved `json_parser` under
`protenix.data.inference`, and tightened `ListValue` so it can no longer take the empty lists
PXDesign's config module declares. None of that is a behaviour change in anything the design
featurizer computes -- it is import paths and one constructor guard -- so re-pointing them is
cheaper and more reproducible than pinning a second 3 GB torch stack.

`pxdbench` is a genuinely absent package. It is stubbed with an object that raises on any
attribute access, so a capture that actually reaches it fails loudly instead of quietly
picking up a wrong default.

Capture-time only: nothing in `tt_bio` imports this, and the gate that consumes the capture
needs neither the shim nor any upstream install.
"""
from __future__ import annotations

import sys
import types
import warnings

warnings.filterwarnings("ignore")

_MOVED = [
    ("ccd", "protenix.data.core.ccd"),
    ("parser", "protenix.data.core.parser"),
    ("featurizer", "protenix.data.core.featurizer"),
    ("filter", "protenix.data.core.filter"),
    ("substructure_perms", "protenix.data.core.substructure_perms"),
    ("json_parser", "protenix.data.inference.json_parser"),
]


class _Poison:
    def __init__(self, name):
        self._n = name

    def __getattr__(self, k):
        raise RuntimeError(f"pxdbench stub touched: {self._n}.{k}")

    def __getitem__(self, k):
        raise RuntimeError(f"pxdbench stub touched: {self._n}[{k!r}]")

    def __call__(self, *a, **k):
        raise RuntimeError(f"pxdbench stub called: {self._n}")


def install():
    """Idempotent. Returns the list of shims actually applied, for the capture's meta."""
    import protenix.data as pd

    applied = []
    for short, full in _MOVED:
        if hasattr(pd, short):
            continue
        try:
            m = __import__(full, fromlist=["*"])
        except ImportError as e:
            print(f"[shim] {short}: {e}", file=sys.stderr)
            continue
        setattr(pd, short, m)
        sys.modules["protenix.data." + short] = m
        applied.append(f"protenix.data.{short} -> {full}")

    for name in ("pxdbench", "pxdbench.pxd_configs", "pxdbench.pxd_configs.eval"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
    sys.modules["pxdbench.pxd_configs.eval"].eval_configs = _Poison("eval_configs")
    sys.modules["pxdbench.pxd_configs"].eval = sys.modules["pxdbench.pxd_configs.eval"]
    sys.modules["pxdbench"].pxd_configs = sys.modules["pxdbench.pxd_configs"]
    applied.append("pxdbench: poison stub")

    from protenix.config import extend_types as et
    if not getattr(et.ListValue, "_pxdesign_empty_ok", False):
        orig = et.ListValue.__init__

        def patched(self, value, dtype=None):
            if value is not None and len(value) == 0:
                self.value, self.dtype = value, (dtype or str)
            else:
                orig(self, value, dtype)

        et.ListValue.__init__ = patched
        et.ListValue._pxdesign_empty_ok = True
        applied.append("protenix.config.ListValue: empty list keeps its declared dtype")
    return applied
