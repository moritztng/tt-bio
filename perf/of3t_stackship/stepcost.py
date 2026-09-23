#!/usr/bin/env python3
"""of3t-stackship: the cost of the exact training default on one OpenFold3 training step.

    stepcost.py --exact on|off --out F.json

`perf/of3t_trainfwd/trainfwd_run.py --arm full`, unchanged: the registered adapter's forward,
the `af3` objective's seeds, one backward, through the shipped tape. `--exact off` wraps it in
`autograd.exact_training(False)`, the public off switch; `--exact on` is the default and opens
nothing. The exact counters are read after the step and written beside its wall time and the
AICLK trainfwd_run samples DURING the work.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    exact = argv[argv.index("--exact") + 1]
    out = Path(argv[argv.index("--out") + 1])
    assert exact in ("on", "off"), exact
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trainfwd"))
    from tt_bio import autograd as ag
    import trainfwd_run

    sys.argv = ["trainfwd_run.py", "--arm", "full", "--out", str(out)]
    with (ag.exact_training(False) if exact == "off" else contextlib.nullcontext()):
        ops = list(ag.exact_training_ops())
        rc = trainfwd_run.main()
    rec = json.loads(out.read_text())
    rec["stackship"] = {"exact": exact, "exact_training_ops": ops,
                        "counters": {"softmax": dict(ag.EXACT_SOFTMAX_STATS),
                                     "layer_norm": dict(ag.EXACT_LAYER_NORM_STATS)}}
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("STEPCOST " + json.dumps({"exact": exact, "step_s": rec.get("step_s"),
                                    "aiclk": rec.get("aiclk_line"),
                                    "counters": rec["stackship"]["counters"]}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
