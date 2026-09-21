#!/usr/bin/env python3
"""An env var read in two modules must carry the SAME default in both (D151).

WHY THIS EXISTS
---------------
`TT_BIO_SOFTMAX_BW_RENORM` is read twice in the composed tree:

    tt_bio/autograd.py:80    SOFTMAX_BW_RENORM  = env_flag("TT_BIO_SOFTMAX_BW_RENORM", False)
    tt_bio/taped_ttnn.py:202 _SOFTMAX_BW_RENORM = os.environ.get("TT_BIO_SOFTMAX_BW_RENORM", "0")

and the two reach DIFFERENT backends. After `of3t-d116` unified the device softmax backward into
`autograd.softmax_bw_inner`, that helper -- used by BOTH `triangle_attention` and
`taped_ttnn._v_softmax` -- branches on autograd's copy, while the HOST float64 backward at
`autograd.py:906` branches on taped_ttnn's. A comment there says *"honoured here so ONE flag
covers both softmax backends"*, and that sentence is what would stop the next reader checking.

Today both default off, so the tree behaves identically whichever is read -- this is a LATENT
defect, and the check is green when it lands. It stops being latent the moment `of3t-d56-renorm`
ships Moritz's ask-9629 decision by flipping ONE of the two to on: the device path would take the
repair and the host float64 path would not, or the reverse, and the flag would read ON in one
place and OFF in the place that runs. That is the half-fix `of3t-d116`'s own docstring warns
about -- *"a repair applied to one of two identical expressions is the kind of half-fix that reads
as fixed"* -- arriving through the back door of a second module.

WHAT IT CHECKS, AND WHY NOT MORE
--------------------------------
For every `TT_BIO_*` variable read at more than one place in `tt_bio/`, the literal defaults must
agree after normalisation (`""`, `"0"`, `"false"`, `"no"`, `"off"`, `False` all mean off).

It does NOT require one read per variable. That version was measured first: 9 of 166 variables are
read more than once and 8 are benign -- the same file reading twice, or three models reading one
shared seed -- so it would be 89 % false positives, which is the check declined at pass 261 for
the same reason. Disagreeing DEFAULTS is the property that can actually make two sites behave
differently, and it is 0 of 166 today.

CPU only. Reads the composed tree it is pointed at.
"""
from __future__ import annotations

import ast
import collections
import pathlib
import sys

OFF = ("", "0", "false", "no", "off")


def _norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "off" if v is False else "on"
    if isinstance(v, str):
        return "off" if v.strip().lower() in OFF else "on:" + v
    return repr(v)


def _is_env_read(fn):
    """`os.environ.get`, `os.getenv`, or the house `env_flag` helper."""
    if isinstance(fn, ast.Attribute) and fn.attr == "get" \
       and isinstance(fn.value, ast.Attribute) and fn.value.attr == "environ":
        return True
    if isinstance(fn, ast.Attribute) and fn.attr == "getenv":
        return True
    return isinstance(fn, ast.Name) and fn.id in ("env_flag", "getenv")


def reads(root: pathlib.Path):
    out = collections.defaultdict(list)
    for f in sorted((root / "tt_bio").rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_env_read(node.func)):
                continue
            if not (node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                continue
            name = node.args[0].value
            if not name.startswith("TT_BIO"):
                continue
            dflt = node.args[1].value if len(node.args) > 1 \
                and isinstance(node.args[1], ast.Constant) else None
            out[name].append((str(f.relative_to(root)), node.lineno, _norm(dflt)))
    return out


def offenders(root: pathlib.Path):
    bad = []
    for name, sites in sorted(reads(root).items()):
        if len(sites) > 1 and len({d for _f, _l, d in sites}) > 1:
            bad.append((name, sites))
    return bad


def main(argv):
    root = pathlib.Path(argv[1] if len(argv) > 1 else ".").resolve()
    bad = offenders(root)

    # Probe: the guard must fire on the exact change that is coming. Take the real tree and flip
    # ONE of the two renorm defaults, the way of3t-d56-renorm's branch does.
    import tempfile
    import shutil
    probe_src = root / "tt_bio" / "taped_ttnn.py"
    if probe_src.is_file():
        with tempfile.TemporaryDirectory() as td:
            t = pathlib.Path(td)
            (t / "tt_bio").mkdir()
            for f in (root / "tt_bio").glob("*.py"):
                shutil.copy2(f, t / "tt_bio" / f.name)
            p = t / "tt_bio" / "taped_ttnn.py"
            txt = p.read_text(errors="replace")
            flipped = txt.replace('os.environ.get("TT_BIO_SOFTMAX_BW_RENORM", "0")',
                                  'os.environ.get("TT_BIO_SOFTMAX_BW_RENORM", "1")')
            if flipped == txt:
                print("BROKEN the probe could not find the read it flips -- the tree moved and "
                      "this guard is no longer testing what it claims", file=sys.stderr)
                return 2
            p.write_text(flipped)
            if not any(n == "TT_BIO_SOFTMAX_BW_RENORM" for n, _s in offenders(t)):
                print("BROKEN flipping one of two defaults does not fire -- this guard proves "
                      "nothing", file=sys.stderr)
                return 2

    if bad:
        for name, sites in bad:
            print("  DRIFT %s is read with disagreeing defaults (D151):" % name)
            for f, l, d in sites:
                print("        %s:%d default=%s" % (f, l, d))
        print("FAIL %d env var(s) whose two readers can disagree; give them one read or one "
              "default" % len(bad))
        return 1

    n = sum(1 for _n, s in reads(root).items() if len(s) > 1)
    print("ok    %d TT_BIO_* var(s) are read in more than one place and every one agrees on its "
          "default (probe fires on flipping one of TT_BIO_SOFTMAX_BW_RENORM's two)" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
