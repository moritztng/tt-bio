"""Census every torch.amp.autocast CONTEXT SITE in the 0.4.3 tree, by what it asks for.

The bf16 arm maps `autocast("cuda", enabled=False)` faithfully (autocast is off either way) but
cannot map `autocast("cuda", dtype=torch.float32)` faithfully, because CPU autocast has no
float32 target. So the census splits exactly along the line that decides how loose the arm's
reading is as a bound on upstream's own bf16 floor.
"""
import ast
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sites = []
for f in sorted(root.rglob("*.py")):
    try:
        tree = ast.parse(f.read_text())
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = None
        if isinstance(fn, ast.Attribute):
            name = fn.attr
        if name != "autocast":
            continue
        kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
        pos = [ast.unparse(a) for a in node.args]
        dtype = kw.get("dtype")
        enabled = kw.get("enabled")
        if enabled == "False":
            kind = "enabled=False (faithful on CPU: autocast off either way)"
        elif dtype in ("torch.float32",):
            kind = "dtype=torch.float32 (NOT faithful on CPU: no fp32 autocast target)"
        elif dtype is not None:
            kind = f"dtype={dtype} (variable)"
        else:
            kind = "plain"
        sites.append({
            "file": str(f.relative_to(root)),
            "line": node.lineno,
            "device": (pos[0] if pos else kw.get("device_type")),
            "dtype": dtype,
            "enabled": enabled,
            "kind": kind,
        })

by_kind = {}
for s in sites:
    by_kind.setdefault(s["kind"], []).append(f"{s['file']}:{s['line']}")
out = {
    "n_context_sites": len(sites),
    "all_name_device_cuda": all(
        (s["device"] or "").strip("'\"") == "cuda" for s in sites),
    "by_kind": {k: {"n": len(v), "sites": v} for k, v in by_kind.items()},
    "sites": sites,
}
print(json.dumps(out, indent=1))
