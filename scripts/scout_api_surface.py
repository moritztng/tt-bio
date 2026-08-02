#!/usr/bin/env python3
"""Scout-only: dump the ttnn symbol surface that tt-bio actually calls, plus the full
top-level/experimental symbol sets, so two ttnn versions can be diffed card-free.

Usage:  python3 scripts/scout_api_surface.py <out.json>
Run once per venv with PYTHONNOUSERSITE=1. No device needed.
"""
import json, re, sys, inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(r"ttnn\.[A-Za-z_][A-Za-z0-9_.]*")

def used_paths():
    seen = set()
    for sub in ("tt_bio", "scripts"):
        for f in (ROOT / sub).rglob("*.py"):
            try:
                txt = f.read_text(errors="ignore")
            except OSError:
                continue
            for m in PAT.findall(txt):
                seen.add(m.rstrip("."))
    return sorted(seen)

def resolve(mod, path):
    obj = mod
    for part in path.split(".")[1:]:
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj

def describe(obj):
    d = {"type": type(obj).__name__}
    try:
        d["sig"] = str(inspect.signature(obj))
    except (TypeError, ValueError):
        d["sig"] = None
    return d

def main():
    import ttnn
    import importlib.metadata as md
    out = {"ttnn_version": md.version("ttnn"), "used": {}, "toplevel": [], "experimental": [], "transformer": []}
    for p in used_paths():
        obj = resolve(ttnn, p)
        out["used"][p] = {"present": False} if obj is None else dict(present=True, **describe(obj))
    out["toplevel"] = sorted(n for n in dir(ttnn) if not n.startswith("_"))
    for sub in ("experimental", "transformer"):
        m = getattr(ttnn, sub, None)
        out[sub] = sorted(n for n in dir(m) if not n.startswith("_")) if m is not None else []
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True))
    missing = [p for p, v in out["used"].items() if not v["present"]]
    print("ttnn {}: {} used paths, {} MISSING".format(out["ttnn_version"], len(out["used"]), len(missing)))
    for p in missing:
        print("  MISSING:", p)

if __name__ == "__main__":
    main()
