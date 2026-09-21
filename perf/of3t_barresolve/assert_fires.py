#!/usr/bin/env python3
"""`trajbar.resolve_ref()` executed, both directions. An assertion nobody ran is not a check.

Positive: call `resolve_ref()` the way `run_arm` now does and show it returns the 0.4.3 tree.
Negative: install 0.5.0 ahead of everything and ask for 0.4.3 back. It must refuse, and the
refusal must NAME the tree that answered -- a guard that raises without saying what it got
sends the next reader looking in the wrong place.

CPU only, no card, no checkpoint load: this exercises the resolution, not the arm.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [p for p in (os.path.join(os.path.dirname(HERE), "of3t_trajbar"),) if p not in sys.path]

import trajbar as TB                                                     # noqa: E402


def main() -> int:
    rp = TB.TW.refpath
    ok = True

    got = TB.resolve_ref()
    import openfold3
    want = os.path.realpath(rp.OF3PKG)
    print(f"positive: resolve_ref() -> {got}")
    if got != want or not os.path.realpath(openfold3.__file__).startswith(want):
        print(f"FAIL positive: expected {want}, openfold3.__file__ = {openfold3.__file__}")
        ok = False
    if TB.REF_TREE is not None:
        print(f"FAIL positive: module REF_TREE is {TB.REF_TREE!r} before run_arm set it")
        ok = False

    # The negative control has to be a tree that EXISTS and is a different openfold3, or it
    # tests nothing: a missing path prepends nothing and the import silently falls through,
    # which is D153 rather than D149.
    wrong = os.path.join(os.path.dirname(want), "of3pkg050")
    if not os.path.isdir(os.path.join(wrong, "openfold3")):
        wrong = "/home/ttuser/of3t-campaign-refs/of3pkg050"
    if not os.path.isdir(os.path.join(wrong, "openfold3")):
        print(f"FAIL negative: no 0.5.0 tree to break it with (looked at {wrong})")
        return 1

    # Fresh interpreter: `openfold3` is already imported in this one, so a path change here
    # would prove nothing about what an import resolves to.
    import subprocess
    code = (
        "import sys, os\n"
        f"sys.path.insert(0, {wrong!r})\n"
        f"sys.path.append({rp.REFDEPS[0]!r})\n"
        f"sys.path.insert(0, {os.path.dirname(rp.__file__)!r})\n"
        "import refpath\n"
        f"print(refpath.assert_resolved({want!r}))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    msg = (r.stdout + r.stderr).strip()
    print(f"negative: rc={r.returncode}\n  {msg.splitlines()[-1] if msg else '(no output)'}")
    if r.returncode == 0:
        print("FAIL negative: assert_resolved accepted a 0.5.0 tree as 0.4.3")
        ok = False
    elif os.path.realpath(wrong) not in msg:
        print(f"FAIL negative: the refusal does not name the tree that answered ({wrong})")
        ok = False

    print("assert_fires: " + ("pass" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
