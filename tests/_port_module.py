"""Load a `scripts/<port>/<name>.py` helper under a name that cannot collide.

Four ports ship a `parity_gate.py` and the suite imports two of them. `sys.path.insert` plus a
bare `import parity_gate` resolves by whichever test module inserted last, which is collection
order, so the second gate to run gets the first one furniture. Load by file path under
`<port>.<name>` instead: unique in `sys.modules`, and no sys.path entry left behind.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def port_module(port: str, name: str):
    key = f"{port}.{name}"
    mod = sys.modules.get(key)
    if mod is None:
        path = _REPO / "scripts" / port / f"{name}.py"
        spec = importlib.util.spec_from_file_location(key, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod
