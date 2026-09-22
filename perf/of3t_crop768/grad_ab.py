#!/usr/bin/env python3
"""PROTOCOL SS3's gradient instrument, run with the dead-value release ON and OFF.

The lever this row builds changes WHICH BUFFERS THE CARD KEEPS and nothing else -- no op is
added or removed on the forward, no accumulation order moves, no dtype changes. That is a
claim about the gradient, so it is tested on the gradient, against the FLOAT64 REFERENCE
`instrument_a_grad.py` already validates (upstream's own TriangleMultiplicationIncoming in
float64, itself checked against central finite differences) rather than against the other
device arm. Two arms compared to a common float64 reference is the comparison the protocol
asks for; two device arms compared to each other would agree on a shared error.

The arm is set by assigning the module flag before the instrument imports anything, not by
an env var: `_tape` reads `DROP_DEAD_VALUES` at call time, so flipping the global is the
whole switch and the instrument runs unmodified.

    grad_ab.py on  --out perf/of3t_crop768/out/grad_a_on.json
"""
from __future__ import annotations

import argparse
import json
import runpy
import shutil
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
    ag.DROP_DEAD_VALUES = a.arm == "on"
    print(f"[grad_ab] DROP_DEAD_VALUES={ag.DROP_DEAD_VALUES}", flush=True)

    rc = 0
    try:
        runpy.run_path(str(INSTRUMENT), run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0)

    src = INSTRUMENT.with_suffix(".json")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, a.out)
    d = json.loads(a.out.read_text())
    print("[grad_ab] arm=%s verdict=%s forward_rel=%s median=%s" % (
        a.arm, d.get("verdict"), d.get("forward_rel"),
        (d.get("summary") or {}).get("median_rel")), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
