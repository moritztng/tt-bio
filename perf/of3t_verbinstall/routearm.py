#!/usr/bin/env python3
"""of3t-verbinstall D2: the site-selector route with RAW serves suppressed.

`ROUTE_HF` reads 0.5605347900452246 against float64 where the verb install reads
0.4179981990834974 -- the arm that makes MORE softmaxes exact is farther from the true
gradient. The census difference is the whole lead: the route serves 1,685 calls the verb
serves none of, and a raw serve is an exact FORWARD with no tape node
(`tt_bio/autograd.py:920-935`), so the Jacobian through it stays whatever the surrounding
region already was.

This is `ROUTE_HF` with exactly one thing changed. A call that reaches the hook with something
that is not a taped `Tensor` gets `ttnn.softmax` back instead of the float64 one; a taped call
is served as before, by the package's own implementation and not a copy of it. Nothing else
moves: same lever set, same env, same boundary, same cotangent.

Two-sided and pre-registered in `PREREGISTERED.md` D2. ~0.4180 means the raw serves carry the
gap; ~0.5605 means they do not and the candidate mechanism dies.
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
        if argv[i] == "--suppress-out":
            stats_out = argv[i + 1]; i += 2; continue
        rest.append(argv[i]); i += 1

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkceiling"))

    import ttnn
    from tt_bio import autograd as ag
    import arm as ceiling_arm

    SUP = {"suppressed_raw": 0, "served_taped": 0}
    _real = ag.host_f64_softmax

    def _no_raw(x, dim=-1):
        """The hook, minus the raw serves. `site_softmax` calls it as `host(x, dim)`."""
        if isinstance(x, ag.Tensor):
            SUP["served_taped"] += 1
            return _real(x, dim)
        SUP["suppressed_raw"] += 1
        return ttnn.softmax(x, dim=dim)

    # Patched on the module, not on the hook slot: `install()` reads this global when it fills
    # the slot, and it is filled once per tape. Patching the slot instead would be undone by
    # the next `install()` and the arm would quietly become ROUTE_HF again.
    ag.host_f64_softmax = _no_raw

    sys.argv = ["arm.py"] + rest
    rc = ceiling_arm.main()

    import tt_bio.tenstorrent as T
    banked = {
        "what": "ROUTE_HF with raw serves suppressed. `suppressed_raw` is the count that "
                "would have been an exact forward with no tape node; if it is 0 this arm is "
                "ROUTE_HF and proves nothing.",
        "counters": SUP,
        "HOST_F64_SOFTMAX_STATS": dict(T.HOST_F64_SOFTMAX_STATS),
        "argv": rest,
    }
    print("SUPPRESS " + json.dumps(banked))
    if stats_out:
        with open(stats_out, "w") as f:
            json.dump(banked, f, indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
