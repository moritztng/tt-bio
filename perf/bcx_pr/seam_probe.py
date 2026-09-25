#!/usr/bin/env python3
"""Does a downstream package need an upstream change to substitute the predictor?

Answers it statically, off BindCraft 2's own source, with no import of jax and no weights. The
question is narrow: both constructions of `AlphaFoldDesignModel` live in `bindcraft/campaign.py`,
and a downstream backend can only take them over by rebinding the module-level name if BOTH sites
compile to a global load. If either one closed over the class, captured it in a default argument
or aliased it locally, rebinding would reach one site and not the other -- design would run on our
predictor while validation ran on the reference trunk, inside one campaign, silently.

Run:  python3 perf/bcx_pr/seam_probe.py <path-to-BindCraft2-checkout>
"""
import ast
import dis
import sys
from pathlib import Path

NAME = "AlphaFoldDesignModel"
PROTOCOL = "DifferentiableProteinPredictor"


def code_objects(code):
    yield code
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            yield from code_objects(const)


def main(root: Path) -> int:
    campaign = root / "bindcraft" / "campaign.py"
    prediction = root / "bindcraft" / "prediction.py"
    trajectory = root / "bindcraft" / "trajectory.py"
    source = campaign.read_text()

    # 1. every construction of the concrete class, anywhere in the package
    sites = []
    for path in sorted((root / "bindcraft").rglob("*.py")):
        if "/af/" in str(path):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == NAME:
                sites.append(f"{path.relative_to(root)}:{node.lineno}")
    print(f"constructions of {NAME}: {len(sites)} -> {', '.join(sites)}")

    # 2. how each one loads the name. LOAD_GLOBAL is the one that a rebinding reaches.
    loads = []
    for code in code_objects(compile(source, str(campaign), "exec")):
        for ins in dis.get_instructions(code):
            if ins.argval == NAME and ins.opname.startswith("LOAD_"):
                loads.append((code.co_name, ins.positions.lineno, ins.opname))
    for where, line, opname in loads:
        print(f"  campaign.py:{line} in {where}() -> {opname}")
    non_global = [entry for entry in loads if entry[2] != "LOAD_GLOBAL"]

    # 3. the contract upstream already wrote
    proto = ast.parse(prediction.read_text())
    methods = []
    for node in ast.walk(proto):
        if isinstance(node, ast.ClassDef) and node.name == PROTOCOL:
            bases = [ast.unparse(base) for base in node.bases]
            methods = [child.name for child in node.body if isinstance(child, ast.FunctionDef)]
            print(f"{PROTOCOL}({', '.join(bases)}) declares {methods}")
    typed = sum(
        1
        for node in ast.walk(ast.parse(trajectory.read_text()))
        if isinstance(node, ast.FunctionDef)
        and any(
            arg.annotation is not None and PROTOCOL in ast.unparse(arg.annotation)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        )
    )
    print(f"trajectory.py signatures typed against the Protocol: {typed}")

    ok = len(sites) == 2 and len(loads) == 2 and not non_global and methods == ["sequence_gradients"]
    print("\nSUBSTITUTABLE WITHOUT AN UPSTREAM CHANGE:", "yes" if ok else "no")
    if non_global:
        print("  a rebinding would MISS:", non_global)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".bc2")))
