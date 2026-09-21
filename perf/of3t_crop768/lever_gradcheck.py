#!/usr/bin/env python3
"""The dead-value release, run over the ops whose backward it actually changes.

`grad_ab.py` compares a triangle-multiplication module against a float64 reference on both
arms and gets the same digits, but the pin probe shows why that is weak evidence: in that
module the release touches three OUTPUT pins and frees no parent at all, so the path that
saves 29 % of the backward high-water is never taken. The ops it IS taken on are the softmax
and attention verbs, and `perf/hallgrad/gradcheck.py` already checks exactly those against a
float64 reference that is itself validated against central finite differences.

So: run that harness unmodified, once per arm, with `_tape` wrapped in a counter. The arm is
only meaningful if `parents_left_unpinned` is positive on and zero off; the gradient result
is only meaningful if the harness passes on both. Both halves are written to the artifact.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TARGET = REPO / "perf" / "hallgrad" / "gradcheck.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("on", "off"))
    ap.add_argument("--cases", default="softmax,triatt,triatt_gated,triatt_chunked,layernorm,chain,fanin")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn as tt
    ag.DROP_DEAD_VALUES = a.arm == "on"

    c = {"tape_calls": 0, "calls_with_reads": 0, "parents_seen": 0,
         "parents_left_unpinned": 0, "outs_left_unpinned": 0}
    inner = ag._tape

    def counting_tape(out_value, parents, make_fn, reads=None):
        out = inner(out_value, parents, make_fn, reads=reads)
        c["tape_calls"] += 1
        if reads is not None:
            c["calls_with_reads"] += 1
        c["parents_seen"] += len(parents)
        c["parents_left_unpinned"] += sum(1 for p in parents if not getattr(p, "pinned", True))
        if isinstance(out, ag.Tensor) and not out.pinned:
            c["outs_left_unpinned"] += 1
        return out

    ag._tape = counting_tape
    tt._tape = counting_tape
    sys.argv = [str(TARGET), "--cases", a.cases]
    rc = 0
    try:
        runpy.run_path(str(TARGET), run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0)
    finally:
        ag._tape = inner
        tt._tape = inner

    c.update(arm=a.arm, drop_dead_values=ag.DROP_DEAD_VALUES, cases=a.cases, gradcheck_rc=rc)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(c, indent=2))
    print("[lever_gradcheck] " + json.dumps(c), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
