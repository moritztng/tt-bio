#!/usr/bin/env python3
"""No module on the OpenFold3 training path may be invisible to `tape()`.

`tape()` installs itself by rebinding the module-global name `ttnn` in every `tt_bio` module
that holds one, walking `sys.modules` at the moment the tape opens. Three things defeat that,
and all three are silent -- the call site receives a raw handle, computes the right forward and
contributes no gradient, and nothing raises, because the proxy's loud refusal is a property of
the proxy and the proxy was never installed:

  A  `import ttnn` inside a function binds the real module into function scope at call time.
     There is no module global for `_swap` to rebind. A2's two sites were this.
  B  `from ttnn import <name>` inside a function binds the verb itself, which `_swap` cannot
     reach even in principle. `tests/test_tape_reach.py` reads `ast.Import` only, so this form
     is not covered there.
  C  a function-local import of a tt-bio module that holds its own `ttnn` global. `_swap` runs
     over `sys.modules` ONCE, when the tape opens; a module first imported inside the block is
     never shimmed and every call site in it runs raw for that tape. `tape()`'s docstring names
     this as "a caller doing something unusual"; on the training path it is a defect, and it is
     only benign today because something else on the path imports the same module at module
     scope first.

Run:
    tape_reach_guard.py                          assert over the working tree
    tape_reach_guard.py --break-control          + the four artifacts that must FAIL it
    tape_reach_guard.py --tree /some/checkout    over another tree, e.g. a pre-fix commit

A31 requires the negative direction for every requirement rather than the positive one only:
each of A, B and C gets an artifact built to violate exactly it, and the guard must refuse each
one and name the rule it refused under. The two real pre-fix files from `5a2efa001^` are the
fourth control, because a rule that has only ever fired on something I wrote myself has tested
my synthetic and not the defect.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# The OF3 training path, as seeds. The closure below follows tt-bio's own imports from here,
# so a module that joins the path next week is covered without editing this list.
SEEDS = [
    "openfold3", "openfold3_fold", "openfold3_trunk", "openfold3_diffusion",
    "openfold3_diffusion_module", "openfold3_confidence", "openfold3_host_prep",
    "openfold3_msa_embedder", "openfold3_template", "openfold3_batch",
    "autograd", "taped_ttnn", "ops",
]


def _module_file(root: pathlib.Path, dotted: str):
    """`root` is the checkout; every module named here lives under `root/tt_bio`."""
    base = root / "tt_bio"
    p = base / (dotted.replace(".", "/") + ".py")
    if p.exists():
        return p
    p = base / dotted.replace(".", "/") / "__init__.py"
    return p if p.exists() else None


def _module_scope(tree):
    """Statements that run at IMPORT time: everything except function and lambda bodies.

    Class bodies, `if TYPE_CHECKING` blocks and `try: import x except ImportError` all execute
    at module scope and are followed; a `def` body does not and is not. Getting this wrong in
    the obvious way -- `ast.walk` over each top-level node -- makes every function-local import
    look like a module-level one, which both widens the path closure to three other models and
    silently satisfies rule C for the very import it is meant to catch.
    """
    out = []
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        out.append(node)
        for f in ("body", "orelse", "finalbody", "handlers"):
            for sub in getattr(node, f, []) or []:
                if isinstance(sub, ast.stmt) or isinstance(sub, ast.ExceptHandler):
                    stack.append(sub)
    return out


def _imports(tree, here: str, nodes=None):
    """tt-bio modules this file imports, as dotted names under `tt_bio`."""
    out = set()
    pkg = here.rsplit(".", 1)[0] if "." in here else ""
    for node in (ast.walk(tree) if nodes is None else nodes):
        if isinstance(node, ast.ImportFrom):
            if node.level:                                   # from .x import y
                base = pkg if node.level == 1 else ".".join(pkg.split(".")[:-(node.level - 1)])
                mod = f"{base}.{node.module}" if node.module else base
                out.add(mod.lstrip("."))
            elif node.module and node.module.startswith("tt_bio"):
                out.add(node.module[len("tt_bio."):])
        elif isinstance(node, ast.Import):
            for al in node.names:
                if al.name.startswith("tt_bio."):
                    out.add(al.name[len("tt_bio."):])
    return {m for m in out if m}


def training_path(root: pathlib.Path):
    """The MODULE-SCOPE import closure of the OF3 training path, inside `tt_bio`.

    Module scope only, and that is the whole point rather than a simplification. `_swap` walks
    `sys.modules`, so the set it can reach is the set that import time has already built, and
    that is exactly the module-scope closure. Following function-local imports too would drag
    in `protenix`, `opendde` and the CLI through deferred imports in functions the OF3 training
    step never calls, and the guard would then be reporting on three other models.
    """
    seen, stack = set(), list(SEEDS)
    while stack:
        m = stack.pop()
        if m in seen or "_vendor" in m:
            continue
        f = _module_file(root, m)
        if f is None:
            continue
        seen.add(m)
        try:
            tree = ast.parse(f.read_text(), str(f))
        except (SyntaxError, UnicodeDecodeError):
            continue
        stack.extend(_imports(tree, m, nodes=_module_scope(tree)))
    return sorted(seen)


def holds_ttnn_global(path: pathlib.Path):
    """Does this module bind `ttnn` at MODULE scope? That is what `_swap` rebinds."""
    try:
        tree = ast.parse(path.read_text(), str(path))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "ttnn" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "ttnn":
            return True
    return False


def module_scope_imports(root: pathlib.Path, modules):
    """Every tt-bio module that SOME module on the path imports at module scope.

    That is the set `_swap` is guaranteed to have seen by the time a tape opens, because
    tt-bio imports its device modules at import time.
    """
    out = set()
    for m in modules:
        f = _module_file(root, m)
        if f is None:
            continue
        try:
            tree = ast.parse(f.read_text(), str(f))
        except (SyntaxError, UnicodeDecodeError):
            continue
        out |= _imports(tree, m, nodes=_module_scope(tree))
    return out


def observed_in_tape(census_paths):
    """The tt-bio modules a census saw execute a ttnn call site INSIDE a tape.

    The static closure cannot answer which of its members runs under a tape: `openfold3_fold`
    imports `protenix` at module scope, `protenix` imports most of the rest, and the deferred
    `import ttnn as _tn` inside `protenix.fold()` is then on the OF3 "path" by import while
    being unreachable from a training step. Measurement settles it -- these are the modules
    that actually ran verbs inside a tape.
    """
    out = set()
    for p in census_paths or []:
        d = _load(p)
        for s in d.get("sites", []):
            if s.get("line_hits") or s.get("taped_calls") or s.get("raw_calls_in_tape"):
                f = s["file"]
                if f.startswith("tt_bio/") and f.endswith(".py"):
                    out.add(f[len("tt_bio/"):-3].replace("/", "."))
    return out


def asserted_set(mods, observed):
    """What the guard FAILS on, as opposed to what it merely lists.

    The nine `openfold3_*` modules are the training path by definition, and a module a census
    watched run a verb inside a tape is on it by measurement. Everything else in the closure
    is reported and not asserted: a deferred import in a function no training step calls is
    not this guard's business, and asserting on it would make the guard a repo-wide style rule
    that the next model port has to argue with.
    """
    return {m for m in mods if m.split(".")[-1].startswith("openfold3") or m in observed}


def offenders(root: pathlib.Path):
    """Every violation of A, B or C on the training path."""
    mods = training_path(root)
    at_module_scope = module_scope_imports(root, mods)
    bad = []
    for m in mods:
        f = _module_file(root, m)
        if f is None:
            continue
        try:
            tree = ast.parse(f.read_text(), str(f))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Import):
                    for al in sub.names:
                        if al.name == "ttnn":
                            bad.append({"rule": "A", "module": m, "line": sub.lineno,
                                        "func": fn.name, "what": f"import ttnn as {al.asname}"
                                        if al.asname else "import ttnn"})
                        elif al.name.startswith("tt_bio."):
                            tgt = al.name[len("tt_bio."):]
                            _local(root, bad, m, sub, fn, tgt, at_module_scope)
                elif isinstance(sub, ast.ImportFrom):
                    if sub.module == "ttnn":
                        bad.append({"rule": "B", "module": m, "line": sub.lineno,
                                    "func": fn.name,
                                    "what": f"from ttnn import "
                                            f"{', '.join(a.name for a in sub.names)}"})
                        continue
                    for tgt in _imports(ast.Module(body=[sub], type_ignores=[]), m):
                        _local(root, bad, m, sub, fn, tgt, at_module_scope)
    return mods, bad


# `tape()` excludes these two from the swap itself, so importing one late changes nothing:
# `taped_ttnn` IS the proxy and `autograd`'s backward closures must call the real verbs.
NEVER_SHIM = {"taped_ttnn", "autograd"}


def _local(root, bad, m, sub, fn, tgt, at_module_scope):
    if tgt in at_module_scope or tgt in NEVER_SHIM or "_vendor" in tgt:
        return
    f = _module_file(root, tgt)
    if f is not None and holds_ttnn_global(f):
        bad.append({"rule": "C", "module": m, "line": sub.lineno, "func": fn.name,
                    "what": f"function-local import of tt_bio.{tgt}, which holds its own "
                            f"ttnn global and is imported at module scope nowhere on the path"})


# ------------------------------------------------------------------------------------------
# the break control
# ------------------------------------------------------------------------------------------

SYNTH = {
    "A": "def f(x):\n    import ttnn\n    return ttnn.add(x, x)\n",
    "B": "def f(x):\n    from ttnn import add\n    return add(x, x)\n",
    "C": "def f(x):\n    from .openfold3_pathcov_synthetic_dev import go\n    return go(x)\n",
}


def _tree_copy(src: pathlib.Path):
    d = pathlib.Path(tempfile.mkdtemp(prefix="pathcov_ctl_"))
    shutil.copytree(src / "tt_bio", d / "tt_bio",
                    ignore=shutil.ignore_patterns("_vendor", "__pycache__"))
    return d


def break_control(root: pathlib.Path):
    """Every rule refuses an artifact built to violate exactly it, and the two real files do."""
    out = []
    for rule, body in SYNTH.items():
        d = _tree_copy(root)
        name = "openfold3_pathcov_synthetic"
        (d / "tt_bio" / f"{name}.py").write_text("import ttnn\n\n\n" + body)
        if rule == "C":
            (d / "tt_bio" / "openfold3_pathcov_synthetic_dev.py").write_text(
                "import ttnn\n\n\ndef go(x):\n    return ttnn.add(x, x)\n")
        # put the offender ON the path: the seed module imports it at module scope
        seed = d / "tt_bio" / "openfold3_fold.py"
        seed.write_text(f"from .{name} import f  # break control\n" + seed.read_text())
        mods_c, bad = offenders(d)
        bad = [b for b in bad if b["module"] in asserted_set(mods_c, set())]
        hit = [b for b in bad if b["rule"] == rule and b["module"] == name]
        out.append({"control": f"synthetic-{rule}", "fired": bool(hit), "found": hit[:2]})
        shutil.rmtree(d, ignore_errors=True)

    # the real one: A2's two files as they were before 5a2efa001 fixed them
    d = _tree_copy(root)
    real = []
    for f in ("openfold3_confidence.py", "openfold3_host_prep.py"):
        try:
            src = subprocess.check_output(
                ["git", "-C", str(root), "show", f"5a2efa001^:tt_bio/{f}"], text=True)
        except subprocess.CalledProcessError as e:
            real.append({"file": f, "error": str(e)})
            continue
        (d / "tt_bio" / f).write_text(src)
        real.append({"file": f, "restored_bytes": len(src)})
    mods_r, bad = offenders(d)
    bad = [b for b in bad if b["module"] in asserted_set(mods_r, set())]
    fired = sorted({b["module"] for b in bad
                    if b["module"] in ("openfold3_confidence", "openfold3_host_prep")})
    out.append({"control": "real-pre-fix-5a2efa001^", "fired": len(fired) == 2,
                "modules_named": fired, "restored": real,
                "offenders": [b for b in bad if b["module"] in
                              ("openfold3_confidence", "openfold3_host_prep")]})
    shutil.rmtree(d, ignore_errors=True)
    return out



def _load(path):
    o = gzip.open if str(path).endswith(".gz") else open
    with o(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=os.getcwd())
    ap.add_argument("--json", default=None)
    ap.add_argument("--break-control", action="store_true")
    ap.add_argument("--census", action="append", default=[],
                    help="a CENSUS_*.json; its in-tape modules join the asserted set")
    a = ap.parse_args()
    root = pathlib.Path(a.tree).resolve()

    observed = observed_in_tape(a.census)
    mods, bad_all = offenders(root)
    asserted = asserted_set(mods, observed)
    bad = [b for b in bad_all if b["module"] in asserted]
    noted = [b for b in bad_all if b["module"] not in asserted]
    # A guard whose universe is empty passes for the wrong reason. The OF3 training path is
    # forty-odd modules; anything under twenty means the closure broke, not that the tree is
    # clean.
    assert len(mods) >= 20, (
        f"the training-path closure found only {len(mods)} tt_bio modules under {root}; "
        f"a clean verdict over an empty universe is not a clean verdict")
    res = {"tree": str(root), "n_modules_on_training_path": len(mods),
           "n_asserted": len(asserted), "observed_in_tape": sorted(observed),
           "census": a.census, "modules": mods, "asserted": sorted(asserted),
           "offenders": bad, "noted_outside_asserted_set": noted, "passed": not bad}
    if a.break_control:
        res["break_control"] = break_control(root)
        res["break_control_passed"] = all(c["fired"] for c in res["break_control"])
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")
    print(f"training path: {len(mods)} tt_bio modules, {len(asserted)} asserted "
          f"({len(observed)} of them measured running a verb inside a tape); "
          f"offenders: {len(bad)}, noted outside the asserted set: {len(noted)}")
    for b in bad:
        print(f"  {b['rule']}  tt_bio/{b['module']}.py:{b['line']} in {b['func']}(): {b['what']}")
    if a.break_control:
        for c in res["break_control"]:
            print(f"  control {c['control']}: {'FIRED' if c['fired'] else 'DID NOT FIRE'}")
    assert not bad, (
        f"{len(bad)} module(s) on the OF3 training path are invisible to tape(): {bad}")
    if a.break_control:
        assert res["break_control_passed"], (
            "a break control did not fire; a guard that cannot fail has tested nothing: "
            f"{res['break_control']}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
