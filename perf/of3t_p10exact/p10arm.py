#!/usr/bin/env python3
"""of3t-p10exact: the model-frame trunk arm, with the layer-norm backward under measurement.

    p10arm.py [--exact softmax,layer_norm] [--probe-out F] [--ln-fp32] [--probe-limit N]
              [--stats-out F] --census-out C --lever none -- ...

`of3t-stackexact/stackarm.py` with one scope added: `lnprobe.ln_probe()`, which replaces the
taped layer-norm verb with one that measures itself. With `--ln-fp32` the same scope also
RETURNS the fp32 composite, which is the fix under test. Without it the arm is bit-identical to
the unprobed rung, so the same run supplies both the baseline `.pt` and the attribution.

The probe and `--exact layer_norm` both own the layer-norm verb, so they are mutually exclusive.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    argv = sys.argv[1:]
    exact, stats_out, probe_out, ln_fp32, probe_limit, rest, i = [], "", "", False, 32, [], 0
    while i < len(argv):
        if argv[i] == "--exact":
            exact = [x for x in argv[i + 1].split(",") if x]; i += 2; continue
        if argv[i] == "--stats-out":
            stats_out = argv[i + 1]; i += 2; continue
        if argv[i] == "--probe-out":
            probe_out = argv[i + 1]; i += 2; continue
        if argv[i] == "--probe-limit":
            probe_limit = int(argv[i + 1]); i += 2; continue
        if argv[i] == "--ln-fp32":
            ln_fp32 = True; i += 1; continue
        rest.append(argv[i]); i += 1
    assert set(exact) <= {"softmax", "layer_norm"}, exact
    probing = bool(probe_out) or ln_fp32
    assert not (probing and "layer_norm" in exact), "probe and exact layer_norm both own the verb"

    cwd = os.getcwd()
    # Written in the order they are searched (D149: a loop of insert(0, p) reverses it).
    sys.path[0:0] = [os.path.join(cwd, p) for p in (
        "perf/of3t_gradients", "perf/of3t_trunkg043", "perf/of3t_bwdaccum",
        "perf/of3t_trunkceiling", "")]
    sys.path.insert(0, HERE)

    from tt_bio import autograd as ag
    import arm as ceiling_arm          # perf/of3t_trunkceiling/arm.py
    import dev_grad                    # the module dev_cot will import and run
    import exactln
    import lnprobe

    seen = {}
    inner = dev_grad.main

    if "layer_norm" in exact:
        def main_with_exact_ln():
            with exactln.exact_layer_norm():
                seen["layer_norm_verb_is_ours"] = ag._TAPED["layer_norm"] is exactln._verb
                return inner()
        dev_grad.main = main_with_exact_ln
    elif probing:
        # Opened INSIDE `dev_grad.main` for exactln's reason: dev_cot installs its own copy of
        # `_taped_layer_norm` at the start of its `main()`, so a scope opened any earlier is
        # displaced by it and reads as inert.
        def main_with_probe():
            with lnprobe.ln_probe(limit=probe_limit, fp32=ln_fp32):
                seen["layer_norm_verb_is_ours"] = ag._TAPED["layer_norm"] is lnprobe._verb
                return inner()
        dev_grad.main = main_with_probe

    sys.argv = ["arm.py"] + rest
    with (ag.exact_softmax() if "softmax" in exact else contextlib.nullcontext()):
        seen["softmax_installed_inside_the_scope"] = ag.exact_softmax_installed()
        rc = ceiling_arm.main()
        sm = dict(ag.EXACT_SOFTMAX_STATS)

    if probe_out:
        lnprobe.dump(probe_out)

    banked = {
        "exact": sorted(exact),
        "ln_probe": {"installed": probing, "returns_fp32": ln_fp32, "limit": probe_limit},
        "installed": seen,
        "counters": {"softmax": sm, "layer_norm": dict(exactln.STATS),
                     "layer_norm_probe": dict(lnprobe.STATS)},
        "why_counters": "softmax: `verb` taped exact forward+Jacobian, `raw` the module-wide "
                        "ttnn.softmax. layer_norm: `verb` taped exact forward, `bw` exact "
                        "backwards taken, `raw` untaped forwards. layer_norm_probe: `bw` "
                        "backwards seen by the probe, `probed` of those measured against "
                        "float64, `fp32_returned` handed back as the fp32 composite.",
        "argv": rest,
    }
    print("STACK_EXACT " + json.dumps(banked))
    if stats_out:
        with open(stats_out, "w") as f:
            json.dump(banked, f, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
