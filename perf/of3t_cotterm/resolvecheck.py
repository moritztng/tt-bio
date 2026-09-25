#!/usr/bin/env python3
"""of3t-cotterm: did the PRE-FIX runs resolve to the trees they meant to? Measured, not argued.

`armapb.py` and `refapb.py` carried the D149 reversing loop until this pass, so three artifacts
already exist that were produced under the reversed order: the float64 reference at padded 384
(`REF_APB_n384.json`), its A/A floor (`FLOOR_AA_REF_APB_N384.json`), and the device arm CAPA
that was on card 1 when the fix landed. Re-running the reference costs 1,266 s and the arm costs
more, so the honest cheap thing is to establish whether the reversal could have changed which
file answered any import -- and to say so with the resolutions read back, not with an argument
about what should have happened.

The method is direct: build both sys.path orders, and for every module either script imports,
ask `importlib` which file each order would pick. A name that answers from the same file under
both orders was not affected by the reversal, whatever the precedence said.

CPU only, no board.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ARM_NS = ("", "perf/of3t_apbleaf", "perf/of3t_trunkg043", "perf/of3t_gradients",
          "perf/of3t_cotterm")
REF_NS = ("", "perf/of3t_apbleaf", "perf/of3t_cotterm")

#: every module name the two scripts and their callees import from these namespaces
NAMES = ("apbmath", "armln", "refln", "dev_grad", "ref_grad", "score", "trees",
         "tt_bio", "openfold3")


def abspaths(ns):
    return [os.path.join(ROOT, p) if p else ROOT for p in ns]


def which(name, path):
    """The file `name` would resolve to with exactly this sys.path, without importing it."""
    saved = sys.path[:]
    sys.path[:] = path
    try:
        spec = importlib.util.find_spec(name)
        return spec.origin if spec else None
    except Exception as exc:                                               # noqa: BLE001
        return f"error: {type(exc).__name__}: {exc}"
    finally:
        sys.path[:] = saved


def compare(ns, tail):
    """Reversed (pre-fix, insert-at-0 in a loop) vs spliced (post-fix), per module."""
    paths = abspaths(ns)
    pre = list(reversed(paths)) + tail        # what the loop actually produced
    post = paths + tail                       # what install() produces
    rows = {}
    for n in NAMES:
        a, b = which(n, pre), which(n, post)
        rows[n] = {"pre_fix": a, "post_fix": b, "same": a == b}
    return {"pre_fix_order": pre[:len(paths)], "post_fix_order": post[:len(paths)],
            "modules": rows,
            "n_differ": sum(1 for r in rows.values() if not r["same"]),
            "differ": [n for n, r in rows.items() if not r["same"]]}


def main() -> int:
    tail = [p for p in sys.path if not p.startswith(ROOT)]
    R = {"what": __doc__.strip().splitlines()[0],
         "host": socket.gethostname(), "board_class": "host CPU only, no board",
         "card": "n/a (CPU)", "aiclk_during": "n/a (CPU)",
         "root": ROOT,
         "armapb": compare(ARM_NS, tail),
         "refapb": compare(REF_NS, tail)}

    # and the live read-back: import tt_bio under the PRE-FIX order and see what answers.
    sys.path[:] = list(reversed(abspaths(ARM_NS))) + tail
    import tt_bio                                                          # noqa: E402
    R["tt_bio_file_under_pre_fix_order"] = tt_bio.__file__
    R["tt_bio_under_worktree"] = os.path.realpath(tt_bio.__file__).startswith(
        os.path.realpath(ROOT))
    # `openfold3` is never on these namespaces at all; both reference scripts put the reference
    # tree on sys.path themselves and refln refuses a tree that is not the one it was given.
    R["openfold3_on_row_namespaces"] = which("openfold3", abspaths(ARM_NS))
    R["verdict"] = ("the reversal changed nothing: every module both scripts import answers from "
                    "the same file under either order"
                    if R["armapb"]["n_differ"] == 0 and R["refapb"]["n_differ"] == 0
                    else "RESOLUTION DIFFERS -- the pre-fix artifacts are against the wrong tree")
    out = os.path.join(HERE, "RESOLVED_COTTERM.json")
    json.dump(R, open(out, "w"), indent=2)
    print(json.dumps(R, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
