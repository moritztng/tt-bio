"""tap_gate.py with its host torch arms memoised to disk.

The af2ig device leg spends ~77 min at qb1's load on two CPU forward passes (float32 and torch
bfloat16) that build the envelope, and ~1 min on the card. Those two arms never touch the device,
so they are the same for every device arm scored off one tree: the shipped arm and the
`--template-host` control share them. This runs tap_gate's own main() unchanged and caches only
the calls with device=False, keyed on every argument that shapes them.

    ENV_CACHE=/path/dir PYTHONPATH=<tree> python3 perf/bcx_land/tap_gate_cached.py <tap_gate args>
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

tree = Path(os.environ["PYTHONPATH"].split(":")[0])
spec = importlib.util.spec_from_file_location("tap_gate", tree / "scripts/af2_port/tap_gate.py")
tg = importlib.util.module_from_spec(spec)
sys.modules["tap_gate"] = tg
spec.loader.exec_module(tg)

cache = Path(os.environ["ENV_CACHE"])
cache.mkdir(parents=True, exist_ok=True)
_run_arm = tg.run_arm


def run_arm(state, feats, prev, **kw):
    if kw.get("device"):
        return _run_arm(state, feats, prev, **kw)
    key = repr(sorted((k, sorted(v) if isinstance(v, set) else v) for k, v in kw.items()))
    path = cache / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".pt")
    if path.exists():
        blob = torch.load(path, weights_only=False)
        assert blob["key"] == key
        taps = tg.Taps(keep=kw.get("keep"))
        taps.values, taps.produced = blob["values"], blob["produced"]
        print(f"# host arm {kw.get('dtype')} from cache {path}", file=sys.stderr)
        return taps, blob["last"]
    taps, last = _run_arm(state, feats, prev, **kw)
    torch.save({"key": key, "values": taps.values, "produced": taps.produced, "last": last},
               str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)
    return taps, last


tg.run_arm = run_arm
sys.argv[0] = str(tree / "scripts/af2_port/tap_gate.py")
raise SystemExit(tg.main())
