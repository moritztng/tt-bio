#!/usr/bin/env python3
"""No NEW OF3T script may put two package trees on `sys.path` without proving which one it got (D149).

WHY THIS EXISTS
---------------
`of3t-trajwide`'s reference side ran openfold3 **0.5.0** for its whole life while its constant, its
docstring and the campaign's GO-condition-3 headline all said 0.4.3. The mechanism is three inserts
at the same index:

    for p in (OF3PKG,) + REFDEPS:          # (of3pkg043, deps, pylibs)
        if p not in sys.path:
            sys.path.insert(1, p)          # <- pylibs goes in LAST and lands FIRST

and `pylibs` carries a whole second `openfold3`. Measured: the checkpoint loads into 0.4.3 at 0
missing / 0 unexpected and into 0.5.0 at 1 missing / 24 unexpected, and D120 puts those two at 4.5
orders apart at this boundary. A named constant is not a resolution -- nothing read
`openfold3.__file__` back.

WHAT IT CHECKS, AND WHY IT IS THIS NARROW
-----------------------------------------
A file under `perf/of3t_*/` may not contain a LOOP whose body calls `sys.path.insert(<constant>, p)`
unless it also READS THE RESOLUTION BACK -- `openfold3.__file__` or `openfold3.__path__` -- in the
same file. Inserting N paths at one fixed index always reverses their order, so the element the
author wrote FIRST, meaning "prefer this tree", ends up LAST.

The first version of this check was broader -- "two or more sys.path entries and no resolution
read" -- and it flagged 18 further files on top of the 5. Every one of them was a separate
`sys.path.insert(0, os.getcwd())` next to one package insert, which is not the defect. A guard
mostly wrong gets ignored (the pass-261 check, 4-of-5 false positives, was declined for exactly
this), so the rule is the MECHANISM and not the nicety. Scoped to `perf/of3t_*` it matches 5 files
and they are the 5 real ones.

What it therefore does NOT catch: a file that hard-codes the wrong order without a loop. Only
reading the resolution back catches that, and that is what AMENDMENT 3 requires of the row.

SHRINK-ONLY RATCHET, and why. The offenders below are frozen because they belong to rows that have
concluded or are mid-flight; failing the compose on them would wedge the campaign on files this
row does not own. A NEW one fails. The list may only get shorter: if a frozen file is repaired and
still listed, that is also a failure, so the ratchet cannot rust.

The fifth trajwide file, `val_ref.py`, was found by this sweep and not by reading four others.

CPU only. Reads the composition it is pointed at.
"""
from __future__ import annotations

import ast
import pathlib
import sys

#: path -> why it is frozen. Shrink-only, and as of pass 272 it is EMPTY: `of3t-trajwide` repaired
#: all five of its files one pass after AMENDMENT 3, so the ratchet now says no of3t script does
#: this at all, which is the strongest state it can be in and the only one worth defending. The
#: row's repair is better than the amendment asked for: `perf/of3t_trajwide/refpath.py` holds one
#: `install()` that appends the dep trees and prepends only the tree under test, and one
#: `assert_resolved()` that imports the package, reads `openfold3.__file__` and refuses with the
#: resolved path -- shared by all five scripts instead of five copies of a fix.
FROZEN = {}

RESOLUTION_READ = ("openfold3.__file__", "openfold3.__path__", "of3.__file__")


def reversing_loop_lines(text):
    """Line numbers of `for` loops whose body inserts onto sys.path at a CONSTANT index."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        for c in ast.walk(node):
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "insert"
                    and isinstance(c.func.value, ast.Attribute) and c.func.value.attr == "path"
                    and c.args and isinstance(c.args[0], ast.Constant)):
                out.append(node.lineno)
                break
    return out


def offenders(root: pathlib.Path):
    out = []
    for f in sorted(root.glob("perf/of3t_*/**/*.py")) + sorted(root.glob("perf/of3t_*/*.py")):
        rel = str(f.relative_to(root))
        text = f.read_text(errors="replace")
        if reversing_loop_lines(text) and not any(r in text for r in RESOLUTION_READ):
            out.append(rel)
    return sorted(set(out))


def main(argv):
    root = pathlib.Path(argv[1] if len(argv) > 1 else ".").resolve()
    found = offenders(root)

    # Probe: the guard must see the shape it exists to catch.
    probe = ("import sys\n"
             "for p in (A, B, C):\n"
             "    sys.path.insert(1, p)\n"
             "from openfold3.core.model.structure.diffusion_module import DiffusionModule\n")
    if not reversing_loop_lines(probe):
        print("BROKEN the probe did not fire -- this ratchet proves nothing", file=sys.stderr)
        return 2
    # A single insert outside a loop is NOT this defect and must not fire.
    if reversing_loop_lines("import sys\nsys.path.insert(0, OF3PKG)\n"):
        print("BROKEN a lone sys.path.insert counts as the reversing pattern", file=sys.stderr)
        return 2
    # Reading the resolution back clears it.
    cleared = probe + "assert openfold3.__file__.startswith(OF3PKG)\n"
    if reversing_loop_lines(cleared) and not any(r in cleared for r in RESOLUTION_READ):
        print("BROKEN a file that reads the resolution back still counts", file=sys.stderr)
        return 2

    new = [f for f in found if f not in FROZEN]
    healed = [f for f in FROZEN if f not in found]
    if new:
        for f in new:
            print("  DRIFT %s inserts several paths at one fixed index, which reverses them, and "
                  "never reads openfold3.__file__ back (D149)" % f)
        print("FAIL %d new file(s) trusting a path constant instead of the resolution" % len(new))
        return 1
    if healed:
        for f in healed:
            print("  DRIFT %s is repaired but still frozen -- remove it from FROZEN (D149)" % f)
        print("FAIL the ratchet has %d stale entr(y/ies); it may only shrink" % len(healed))
        return 1
    if not FROZEN:
        print("ok    NO of3t file inserts paths at a fixed index in a loop without reading the "
              "resolution back -- the D149 ratchet is empty (probe fires, a lone insert does not); "
              "a new one fails")
    else:
        print("ok    %d of3t file(s) insert paths at a fixed index in a loop without reading the "
              "resolution back, all %d frozen (probe fires, a lone insert does not); a new one "
              "fails" % (len(found), len(FROZEN)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
