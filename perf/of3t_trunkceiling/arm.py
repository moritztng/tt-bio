#!/usr/bin/env python3
"""of3t-trunkceiling: the trunk gradient arm plus the RUNTIME lever census.

This does not rebuild the producer. It runs `of3t_bwdaccum/dev_cot.py`'s main() -- which runs
`of3t_trunkg043/dev_grad.py`'s main() under its LayerNorm-backward patches -- and adds one
thing the campaign has never had in the same artifact as a reading: a count of how many times
each accuracy lever actually fired.

A lever can fire and be inert, and a lever that never fires looks the same in the output
(memory `a-lever-can-fire-and-be-inert`, D121). No comparison of tensors can tell them apart,
so the counters are taken from the package's own `*_STATS` dicts, which are bumped at the call
and not at the construction. Two of them separate the cases explicitly: HOST_F64_SOFTMAX_STATS
counts `refused` for a call that reached the site with the selector ON and no tape to serve it,
and SOFTMAX_BW_RENORM_STATS counts `declined` for the branch evaluated and not taken.

Everything after `--` goes to dev_cot/dev_grad unchanged.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    census_out = ""
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--census-out":
            census_out = argv[i + 1]; i += 2; continue
        rest.append(argv[i]); i += 1

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_bwdaccum"))
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_gradients"))

    import tt_bio.tenstorrent as T
    from tt_bio import autograd as ag

    # Two levers carry no counter of their own, so wrap them. Both are module-global lookups
    # from inside the methods that use them, which is why patching the module attribute
    # reaches the call and not just the name.
    CNT = {"accurate_softmax_chain": 0, "softmax_ckc_calls": 0, "softmax_ckc_precise": 0}
    _real_acc = T._accurate_softmax

    def _acc(*a, **k):
        CNT["accurate_softmax_chain"] += 1
        return _real_acc(*a, **k)

    T._accurate_softmax = _acc

    _real_ckc = T.softmax_ckc

    def _ckc(token, default=False):
        out = _real_ckc(token, default)
        CNT["softmax_ckc_calls"] += 1
        CNT["softmax_ckc_precise"] += int(out is not None)
        return out

    T.softmax_ckc = _ckc

    # The site selectors, recorded as the answers the construction sites will get. This is the
    # config half of the census; the counters above are the runtime half, and the row needs
    # both because a selector can answer True at a site the model never constructs.
    sites = {}
    for fn, var, toks in (
        (T.host_f64_softmax_site, "TT_BIO_HOST_F64_SOFTMAX_AB",
         ("pairformer", "openfold3.trunk", "miniformer")),
        (T.accurate_softmax_site, "TT_BIO_ACCURATE_SOFTMAX_AB",
         ("openfold3.trunk", "pairformer")),
        (T.softmax_precise_site, "TT_BIO_SOFTMAX_PRECISE_AB",
         ("pairformer", "openfold3.trunk")),
        (T.triatt_sdpa_hifi_site, "TT_BIO_TRIATT_SDPA_HIFI_AB", ("openfold3.trunk",)),
        (T.sdpa_ragged_pad_site, "TT_BIO_SDPA_RAGGED_PAD_AB", ("openfold3.trunk",)),
    ):
        for t in toks:
            sites[f"{var}:{t}"] = {"off_default": fn(t, False), "on_default": fn(t, True),
                                   "env": os.environ.get(var, "<unset>")}

    import dev_cot
    sys.argv = ["dev_cot.py"] + rest
    rc = dev_cot.main()

    stats = {n: dict(v) for n, v in sorted(vars(T).items())
             if n.endswith("_STATS") and isinstance(v, dict)}
    stats["SOFTMAX_BW_RENORM_STATS"] = dict(ag.SOFTMAX_BW_RENORM_STATS)
    census = {
        "what": "the runtime lever census for this arm: counts taken at the call, not at the "
                "construction. A lever with 0 in every field did not reach the trunk.",
        "argv": rest,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("TT_BIO_")},
        "site_selectors": sites,
        "wrapped_counters": CNT,
        "package_stats": stats,
        "renorm_flag": bool(ag.SOFTMAX_BW_RENORM),
        "trunk_math_fidelity": T._TRUNK_MATH_FIDELITY,
    }
    print("CENSUS " + json.dumps(census))
    if census_out:
        with open(census_out, "w") as f:
            json.dump(census, f, indent=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
