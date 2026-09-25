#!/usr/bin/env python3
"""of3t-stackship: the model-frame trunk arm through the SHIPPED tape, no lever.

    shipped.py [--device-ops] [--stats-out F] --census-out C --lever none -- ...

`perf/of3t_stackexact/stackarm.py` with every lever removed. It opens no exact scope, patches
no verb and sets no environment variable: whatever runs exact runs because `tape()` and
`backward()` open it. It only READS the package's counters afterwards. `--device-ops` wraps
the run in `autograd.exact_training(False)`, the public off switch, which must reproduce
of3t-stackexact's SHIP_A.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    device_ops, stats_out, rest, i = False, "", [], 0
    while i < len(argv):
        if argv[i] == "--device-ops":
            device_ops = True; i += 1; continue
        if argv[i] == "--stats-out":
            stats_out = argv[i + 1]; i += 2; continue
        rest.append(argv[i]); i += 1

    cwd = os.getcwd()
    # stackarm.py's search order, unchanged (D149).
    sys.path[0:0] = [os.path.join(cwd, p) for p in (
        "perf/of3t_gradients", "perf/of3t_trunkg043", "perf/of3t_bwdaccum",
        "perf/of3t_trunkceiling", "")]

    from tt_bio import autograd as ag
    import arm as ceiling_arm          # perf/of3t_trunkceiling/arm.py

    assert not os.environ.get("TT_BIO_SOFTMAX_BW_RENORM"), "no env var on the shipped arm"
    before = {"softmax": ag.exact_softmax_installed(), "layer_norm": ag.exact_layer_norm_installed()}
    sys.argv = ["arm.py"] + rest
    with (ag.exact_training(False) if device_ops else contextlib.nullcontext()):
        exact_ops = list(ag.exact_training_ops())
        rc = ceiling_arm.main()
    after = {"softmax": ag.exact_softmax_installed(), "layer_norm": ag.exact_layer_norm_installed()}

    banked = {
        "arm": "device_ops" if device_ops else "shipped",
        "exact_training_ops": exact_ops,
        "installed_outside_the_tape": {"before": before, "after": after},
        "counters": {"softmax": dict(ag.EXACT_SOFTMAX_STATS),
                     "layer_norm": dict(ag.EXACT_LAYER_NORM_STATS)},
        "why_counters": "counted at the call inside tt_bio.autograd. softmax: `verb` taped exact "
                        "forward+Jacobian, `raw` ttnn.softmax. layer_norm: `verb` taped exact "
                        "forward, `bw` exact backwards, `raw` untaped forwards. All zero on the "
                        "device-ops arm is the off switch working.",
        "tt_bio_file": ag.__file__,
        "argv": rest,
    }
    print("STACK_SHIP " + json.dumps(banked))
    if stats_out:
        with open(stats_out, "w") as f:
            json.dump(banked, f, indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
