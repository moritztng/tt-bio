#!/usr/bin/env python3
"""of3t-modelever: the model-frame trunk arm, with or without the package's exact softmax.

The model-frame arm the clause is scored on is `dev_cot.py --lever none` under
`TT_BIO_SOFTMAX_BW_RENORM=1`, driven by the graph-cut-external cotangent `cot_external.pt`
(of3t-recut, `DEV_RENORM_MODEL_N384_EXTERNAL.json`). This runs exactly that, through
of3t-trunkceiling's `arm.py` so the runtime lever census is in the same artifact, and with
`--exact` it runs it inside `tt_bio.autograd.exact_softmax()` and nothing else changes.

Both arms go through this one file so that the A/A pair and the lever pair differ by the
`with` block alone. The step-wide scope is the one of3t-verbinstall shipped: the chunked
triangle-attention backward recomputes its probabilities after the tape closes, so a lever
scoped to the tape would leave those on the card.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    exact = False
    stats_out = ""
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--exact":
            exact = True; i += 1; continue
        if argv[i] == "--exact-softmax-out":
            stats_out = argv[i + 1]; i += 2; continue
        rest.append(argv[i]); i += 1

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkceiling"))

    from tt_bio import autograd as ag
    import arm as ceiling_arm          # perf/of3t_trunkceiling/arm.py

    sys.argv = ["arm.py"] + rest
    with (ag.exact_softmax() if exact else contextlib.nullcontext()):
        installed = ag.exact_softmax_installed()
        rc = ceiling_arm.main()
        stats = dict(ag.EXACT_SOFTMAX_STATS)

    banked = {
        "exact_softmax": exact,
        "installed_inside_the_scope": installed,
        "installed_from": "tt_bio.autograd.exact_softmax()" if exact else None,
        "counters": stats,
        "why_counters": "`verb` is the taped verb (exact forward and Jacobian); `raw` is the "
                        "module-wide ttnn.softmax, the only reach into "
                        "triangle_attention._scores. Both 0 on the control arm is the check "
                        "that the control ran without the lever.",
        "argv": rest,
    }
    print("EXACT_SOFTMAX " + json.dumps(banked))
    if stats_out:
        with open(stats_out, "w") as f:
            json.dump(banked, f, indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
