#!/usr/bin/env python3
"""of3t-verbinstall: the `ceiling_hf3` arm again, with the softmax installed FROM THE PACKAGE.

`perf/of3t_bwdaccum/dev_cot.py --lever ceiling_hf3` reads 0.4175214198121818 against the
float64 reference at crop 384. It gets there by rewriting `taped_ttnn._VERBS` and rebinding
`ttnn.softmax` from inside a perf script, so the best number in the campaign belongs to a
harness patch and not to anything a caller can run.

This runs the same frame with `--lever all` -- the fp32 LayerNorm-backward islands, unchanged
-- and takes the softmax from `tt_bio.autograd.exact_softmax()` instead. Everything else is
of3t-trunkceiling's own `arm.py`, imported rather than copied, so the runtime lever census and
the producer stay the ones that made the reading being reproduced.

A reproduction, so a difference is the finding. The reach counters are banked here because
`dev_cot.py` only prints its module-wide `SMRAW` to stdout, and reach is the whole argument the
module-wide half has: the verb alone leaves `triangle_attention._scores` on the card.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    stats_out = ""
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--exact-softmax-out":
            stats_out = argv[i + 1]; i += 2; continue
        rest.append(argv[i]); i += 1

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkceiling"))

    from tt_bio import autograd as ag
    import arm as ceiling_arm          # perf/of3t_trunkceiling/arm.py

    sys.argv = ["arm.py"] + rest
    # Wider than the tape on purpose: `triangle_attention` recomputes its probabilities in the
    # backward and a checkpointed block reruns its forward, both after the tape block has
    # closed. A lever scoped to the tape would leave every one of those on the card.
    with ag.exact_softmax():
        rc = ceiling_arm.main()
        stats = dict(ag.EXACT_SOFTMAX_STATS)

    banked = {
        "what": "the reach of the package's exact softmax, counted at the call. `verb` is the "
                "taped verb (exact forward AND exact Jacobian); `raw` is the module-wide "
                "`ttnn.softmax`, which is the only thing that reaches "
                "`autograd.triangle_attention._scores` in the forward and in the chunked "
                "backward's recompute. A zero in `raw` means the pair track ran on the card.",
        "installed_from": "tt_bio.autograd.exact_softmax() -> install(exact_softmax=True)",
        "counters": stats,
        "argv": rest,
    }
    print("EXACT_SOFTMAX " + json.dumps(banked))
    if stats_out:
        with open(stats_out, "w") as f:
            json.dump(banked, f, indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
