#!/usr/bin/env python3
"""Does the dead-value release actually FIRE inside the module the gradient A/B measures?

Two arms of `grad_ab.py` agreeing to every digit is the result the lever predicts, and it is
also what a lever that never fired would print. So count, in the same process that runs the
instrument: how many `_tape` calls name their reads, and how many parents the release leaves
unpinned. On, that count must be positive; off, it must be zero. Without this the A/B is a
negative control on nothing.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

INSTRUMENT = REPO / "perf" / "of3t_equivalence" / "instrument_a_grad.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("on", "off"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn as tt
    ag.DROP_DEAD_VALUES = a.arm == "on"

    c = {"tape_calls": 0, "calls_with_reads": 0, "parents_seen": 0, "parents_left_unpinned": 0,
         "outs_left_unpinned": 0}
    inner = ag._tape

    def counting_tape(out_value, parents, make_fn, reads=None):
        out = inner(out_value, parents, make_fn, reads=reads)
        c["tape_calls"] += 1
        if reads is not None:
            c["calls_with_reads"] += 1
        c["parents_seen"] += len(parents)
        for p in parents:
            if not getattr(p, "pinned", True):
                c["parents_left_unpinned"] += 1
        if isinstance(out, ag.Tensor) and not out.pinned:
            c["outs_left_unpinned"] += 1
        return out

    ag._tape = counting_tape
    tt._tape = counting_tape
    try:
        runpy.run_path(str(INSTRUMENT), run_name="__main__")
    except SystemExit:
        pass
    finally:
        ag._tape = inner
        tt._tape = inner

    c["arm"] = a.arm
    c["drop_dead_values"] = ag.DROP_DEAD_VALUES
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(c, indent=2))
    print("[probe] " + json.dumps(c), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
