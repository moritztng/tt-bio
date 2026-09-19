#!/usr/bin/env python3
"""Which of the shipped diffusion module's calls a tape can reach, counted rather than assumed.

`ptx-fastpath` owns the same question for the pairformer and quotes it as "0 of 29 routed".
The diffusion module is a different body of code and gets its own count, because the
precision answer is only actionable at the sites a tape can actually attach to.

Routed = dispatched through `tt_bio.ops` (`ops.linear` / `ops.layer_norm`, including the
`_lin` / `_ln` wrappers in `protenix.py` that call them), which is where `tt_bio.autograd`
installs its hook. Unrouted = a bare `ttnn.` call, which the hook cannot see.
"""
import ast, sys
from pathlib import Path
from collections import Counter

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SRC = REPO / "tt_bio" / "protenix.py"

# The diffusion module's own body: `class DiffusionModule` and the two shared primitives the
# token DiT instantiates, which live in tenstorrent.py.
tree = ast.parse(SRC.read_text())
cls = next(n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == "DiffusionModule")

ROUTED_WRAPPERS = {"_lin", "_ln", "_ln_dit", "_lin_nb"}


def name_of(node):
    f = node.func
    if isinstance(f, ast.Attribute):
        base = f.value
        if isinstance(base, ast.Name):
            return f"{base.id}.{f.attr}"
        if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            return f"{base.value.id}.{base.attr}.{f.attr}"
        return f".{f.attr}"
    if isinstance(f, ast.Name):
        return f.id
    return "<expr>"


calls = Counter()
for n in ast.walk(cls):
    if isinstance(n, ast.Call):
        calls[name_of(n)] += 1

routed = {k: v for k, v in calls.items()
          if k.split(".")[-1] in ROUTED_WRAPPERS or k.startswith("ops.")}
ttnn_calls = {k: v for k, v in calls.items() if k.startswith("ttnn.")}
MATH = {"linear", "matmul", "add", "multiply", "subtract", "sigmoid", "relu", "silu", "gelu",
        "softmax", "layer_norm", "rsqrt", "sqrt", "exp", "sum", "mean", "div", "typecast",
        "transpose", "permute", "concat", "reshape", "pad", "slice"}
math_ttnn = {k: v for k, v in ttnn_calls.items() if k.split(".")[-1] in MATH}
inplace = {k: v for k, v in ttnn_calls.items() if k.split(".")[-1].endswith("_")}
dealloc = {k: v for k, v in ttnn_calls.items() if k.split(".")[-1] == "deallocate"}

print(f"DiffusionModule, {SRC}:{cls.lineno}-{max(getattr(x,'end_lineno',0) or 0 for x in ast.walk(cls))}")
print(f"  routed through tt_bio.ops (hook can see):   {sum(routed.values()):4d}  {dict(sorted(routed.items()))}")
print(f"  bare ttnn math calls (hook cannot see):     {sum(math_ttnn.values()):4d}")
for k, v in sorted(math_ttnn.items(), key=lambda x: -x[1]):
    print(f"      {k:26s} {v}")
print(f"  in-place ttnn ops (destroy their input):    {sum(inplace.values()):4d}  {dict(inplace)}")
print(f"  ttnn.deallocate (free a live activation):   {sum(dealloc.values()):4d}")
total = sum(routed.values()) + sum(math_ttnn.values())
print(f"  ROUTED FRACTION: {sum(routed.values())}/{total} = "
      f"{100*sum(routed.values())/total:.1f} %")
