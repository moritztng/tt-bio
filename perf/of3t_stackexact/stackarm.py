#!/usr/bin/env python3
"""of3t-stackexact: the model-frame trunk arm with a chosen set of components made exact.

    stackarm.py [--exact softmax,layer_norm] [--stats-out F] --census-out C --lever none -- ...

of3t-modelever's `modelarm.py`, generalised from one lever to a set, so every rung of the ladder
goes through one file and two rungs differ by the scopes they open and nothing else.

  softmax     `tt_bio.autograd.exact_softmax()`, of3t-modelever's EXACT arm unchanged.
  layer_norm  `exactln.exact_layer_norm()`, opened INSIDE `dev_grad.main`. dev_cot installs its
              own copy of `_taped_layer_norm` at the start of its `main()`, so a scope opened
              any earlier would be displaced by it and read as inert. dev_cot's LN counter is
              the check: it reads 0 backward calls when this scope holds the verb.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    argv = sys.argv[1:]
    exact, stats_out, rest, i = [], "", [], 0
    while i < len(argv):
        if argv[i] == "--exact":
            exact = [x for x in argv[i + 1].split(",") if x]; i += 2; continue
        if argv[i] == "--stats-out":
            stats_out = argv[i + 1]; i += 2; continue
        rest.append(argv[i]); i += 1
    assert set(exact) <= {"softmax", "layer_norm"}, exact

    cwd = os.getcwd()
    for p in ("", "perf/of3t_trunkceiling", "perf/of3t_bwdaccum", "perf/of3t_trunkg043",
              "perf/of3t_gradients"):
        sys.path.insert(0, os.path.join(cwd, p))
    sys.path.insert(0, HERE)

    from tt_bio import autograd as ag
    import arm as ceiling_arm          # perf/of3t_trunkceiling/arm.py
    import dev_grad                    # the module dev_cot will import and run
    import exactln

    seen = {}
    if "layer_norm" in exact:
        inner = dev_grad.main

        def main_with_exact_ln():
            with exactln.exact_layer_norm():
                seen["layer_norm_verb_is_ours"] = ag._TAPED["layer_norm"] is exactln._verb
                return inner()
        dev_grad.main = main_with_exact_ln

    sys.argv = ["arm.py"] + rest
    with (ag.exact_softmax() if "softmax" in exact else contextlib.nullcontext()):
        seen["softmax_installed_inside_the_scope"] = ag.exact_softmax_installed()
        rc = ceiling_arm.main()
        sm = dict(ag.EXACT_SOFTMAX_STATS)

    banked = {
        "exact": sorted(exact),
        "installed": seen,
        "counters": {"softmax": sm, "layer_norm": dict(exactln.STATS)},
        "why_counters": "softmax: `verb` taped exact forward+Jacobian, `raw` the module-wide "
                        "ttnn.softmax. layer_norm: `verb` taped exact forward, `bw` exact "
                        "backwards taken, `raw` untaped forwards. All zero on a rung that does "
                        "not open the scope is the check that it ran without it.",
        "argv": rest,
    }
    print("STACK_EXACT " + json.dumps(banked))
    if stats_out:
        with open(stats_out, "w") as f:
            json.dump(banked, f, indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
