#!/usr/bin/env python3
"""of3t-p10exact: the model-frame trunk arm, with the layer-norm backward under measurement.

    p10arm.py [--device-only] [--exact softmax,layer_norm] [--probe-out F] [--ln-fp32]
              [--probe-limit N] [--sm-fp32] [--stats-out F] --census-out C --lever none -- ...

--device-only wraps the whole arm in `autograd.exact_training(False)`, and WITHOUT IT THERE IS
NO DEVICE-ONLY RUNG on current main. `_EXACT_TRAINING = [True]` (autograd.py:1633) is the
default, so an arm that opens no exact scope still runs the host float64 softmax and layer norm:
measured 2026-09-26, `--exact` empty gave EXACT_SOFTMAX_STATS verb=288 raw=144 over 21,856,518,144
elements, `layer_norm_backward_reach` 0 backwards on the taped verb, 1541 s against the ladder's
170 s, and a clause of 0.15017948190926872 (x_bar 0.98737) which is the softmax+layer_norm rung,
not the none rung. of3t-stackexact's ladder was taken before that default reached this path, so
its `none` rung does not reproduce from here without this flag.

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
    device_only = False
    sm_fp32 = False
    while i < len(argv):
        if argv[i] == "--device-only":
            device_only = True; i += 1; continue
        if argv[i] == "--exact":
            exact = [x for x in argv[i + 1].split(",") if x]; i += 2; continue
        if argv[i] == "--stats-out":
            stats_out = argv[i + 1]; i += 2; continue
        if argv[i] == "--probe-out":
            probe_out = argv[i + 1]; i += 2; continue
        if argv[i] == "--probe-limit":
            probe_limit = int(argv[i + 1]); i += 2; continue
        if argv[i] == "--sm-fp32":
            sm_fp32 = True; i += 1; continue
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
    assert not (device_only and exact), "--device-only and --exact are opposite instruments"
    import arm as ceiling_arm          # perf/of3t_trunkceiling/arm.py
    import dev_grad                    # the module dev_cot will import and run
    import exactln
    import lnprobe
    import smbw32

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
    with (ag.exact_training(False) if device_only else contextlib.nullcontext()):
        seen["exact_training_ops_in_scope"] = [str(o) for o in ag.exact_training_ops()]
        with (ag.exact_softmax() if "softmax" in exact else contextlib.nullcontext()):
            seen["softmax_installed_inside_the_scope"] = ag.exact_softmax_installed()
            with (smbw32.softmax_bw_fp32() if sm_fp32 else contextlib.nullcontext()):
                seen["softmax_bw_fp32_installed"] = smbw32.STATS["installed"]
                rc = ceiling_arm.main()
            sm = dict(ag.EXACT_SOFTMAX_STATS)
            ln_stats = dict(ag.EXACT_LAYER_NORM_STATS)

    if probe_out:
        lnprobe.dump(probe_out)

    banked = {
        "exact": sorted(exact),
        "device_only": device_only,
        "sm_bw_fp32": dict(smbw32.STATS),
        "pkg_softmax_bw_fp32": {"flag": bool(getattr(ag, "SOFTMAX_BW_FP32", False)),
                                **dict(getattr(ag, "SOFTMAX_BW_FP32_STATS", {}))},
        "pkg_softmax_bw_renorm": {"flag": bool(ag.SOFTMAX_BW_RENORM),
                                  **dict(ag.SOFTMAX_BW_RENORM_STATS)},
        "ln_probe": {"installed": probing, "returns_fp32": ln_fp32, "limit": probe_limit},
        "installed": seen,
        "counters": {"EXACT_SOFTMAX_STATS": sm, "EXACT_LAYER_NORM_STATS": ln_stats,
                     "exactln_scope": dict(exactln.STATS),
                     "layer_norm_probe": dict(lnprobe.STATS)},
        "why_counters": "EXACT_SOFTMAX_STATS / EXACT_LAYER_NORM_STATS are the package's own, "
                        "and they are what proves which path ran: both all-zero is the only "
                        "proof of a DEVICE-ONLY arm, because _EXACT_TRAINING defaults True. "
                        "exactln_scope counts the ladder's host float64 layer norm. "
                        "layer_norm_probe: `bw` backwards seen by the probe, `probed` of those "
                        "measured against float64, `fp32_returned` handed back as fp32.",
        "argv": rest,
    }
    print("STACK_EXACT " + json.dumps(banked))
    if stats_out:
        with open(stats_out, "w") as f:
            json.dump(banked, f, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
