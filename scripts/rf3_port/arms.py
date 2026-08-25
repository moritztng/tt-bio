#!/usr/bin/env python3
"""RF3 triangle-attention arms: one place that knows which flags select which route.

`TriangleAttention._attend_heads` (tt_bio/tenstorrent.py) dispatches on two flags, and the
nesting is what defines the arms:

    if _FP32_SOFTMAX or self.fp32_softmax:
        if _TRIATT_FUSED_HIFI:  o = _tri_att_sdpa_hifi(...)   # fused SDPA at HiFi4 + fp32_dest_acc
        if o is None:           o = _fp32_softmax_attention(...)   # materialised fp32 softmax
    else:                       o = _tri_att_sdpa(...)        # fused SDPA at the OP DEFAULT config

`_TRIATT_FUSED_HIFI` is read only inside the branch that `fp32_softmax=False` skips, so a3
("both") is a1 by construction, not a fourth route. It is scored anyway, as a byte-identity
proof against a1 rather than as an independent arm.

`fp32_softmax` and `accurate_softmax` live in RF3s own PAIRFORMER_FLAGS, so they are
model-local. `TT_BIO_TRIATT_FUSED_HIFI` is a global env flag on the shared tenstorrent.py, so
a2 would hit five other models if it were ever defaulted on.

Apply BEFORE `rf3_model.load`: the flags are read at construction, not at call time.
"""
from __future__ import annotations

ARMS = {
    "a0": {},
    "a1": {"fp32_softmax": False},
    "a2": {"triatt_fused_hifi": True},
    "a3": {"fp32_softmax": False, "triatt_fused_hifi": True},
    "a4": {"accurate_softmax": False},
    # a5/a6: a1's route -- the fused kernel at the op default -- with the op default's compute
    # config replaced. a1's error against fp64 is 1.12-2.38x worse than the materialised path in
    # `rel_perp`, the per-channel direction component, and `fp32_dest_acc` owns that component
    # (perf/fused_sdpa/errstruct_rf3_512.json). a5 turns it on and lifts fidelity with it; a6 is a5
    # with math_approx left ON, which isolates the claim that math_approx touches only the gain
    # component and so costs the fold nothing.
    "a5": {"fp32_softmax": False, "sdpa_ckc": "HiFi4,0,1"},
    "a6": {"fp32_softmax": False, "sdpa_ckc": "HiFi4,1,1"},
    # a7/a8/a9: the same three knobs a5 bundles, one at a time off the op default. The op default
    # is (HiFi2, math_approx ON, no fp32_dest_acc) and a5 moves all three at once, so a5 alone
    # cannot say which one paid for what -- an earlier pass conflated exactly this
    # (`rf3-fused-sdpa-fidelity-conflation-and-tiny-fixture-blind-spot`). One-at-a-time plus the
    # bundle is what makes the answer attributable, and a6 (bundle minus math_approx) is the one
    # interaction term worth a row on its own.
    "a7": {"fp32_softmax": False, "sdpa_ckc": "HiFi2,1,1"},   # fp32_dest_acc alone
    "a8": {"fp32_softmax": False, "sdpa_ckc": "HiFi2,0,0"},   # math_approx off alone
    "a9": {"fp32_softmax": False, "sdpa_ckc": "HiFi4,1,0"},   # HiFi4 alone
    # a10 is a5 asked for the way it would SHIP: the per-site `tri_att_sdpa_hifi` selector rather
    # than the process-global `_CKC_OVERRIDE`. Both end at the same `ckc` tuple in
    # `triatt_sdpa.sdpa`, so a10 must be bit-identical to a5; if it is not, the wiring is wrong and
    # no a5 number describes what a flip would do.
    "a10": {"fp32_softmax": False, "sdpa_hifi_site": True},
}

ROUTE = {
    # a0 overrides nothing, so its route is whatever the tree ships. That is _tri_att_sdpa since
    # the fast arm became the default; it was _fp32_softmax_attention before the flip. This string
    # is stamped into every result JSON, so do not hardcode a route name here again -- the
    # resolved flags below it are the ones that cannot go stale.
    "a0": "whatever the tree ships, no override applied -- read fp32_softmax under resolved",
    "a1": "_tri_att_sdpa (fused SDPA at the op-default config: HiFi2, math_approx, no fp32_dest_acc)",
    "a2": "_tri_att_sdpa_hifi (fused SDPA at HiFi4 + fp32_dest_acc) above 128 tokens, "
          "_fp32_softmax_attention below",
    "a3": "identical to a1: _TRIATT_FUSED_HIFI is read only inside the fp32_softmax branch",
    "a4": "_fp32_softmax_attention with ttnn.softmax instead of the 5-op reduction chain",
    "a5": "_tri_att_sdpa at HiFi4 + math_approx OFF + fp32_dest_acc ON (a1's route, a1's chunks)",
    "a6": "_tri_att_sdpa at HiFi4 + math_approx ON + fp32_dest_acc ON (a5 minus the approx knob)",
    "a7": "_tri_att_sdpa at HiFi2 + math_approx ON + fp32_dest_acc ON (fp32_dest_acc alone)",
    "a8": "_tri_att_sdpa at HiFi2 + math_approx OFF + no fp32_dest_acc (math_approx alone)",
    "a9": "_tri_att_sdpa at HiFi4 + math_approx ON + no fp32_dest_acc (fidelity alone)",
    "a10": "_tri_att_sdpa at _TRIATT_FUSED_HIFI_CKC through the per-site `tri_att_sdpa_hifi` "
           "selector; the same arithmetic as a5, reached the way a flip would reach it",
}

SCOPE = {
    "a0": "model-local (shipped default)",
    "a1": "model-local (PAIRFORMER_FLAGS)",
    "a2": "GLOBAL (TT_BIO_TRIATT_FUSED_HIFI on the shared tenstorrent.py)",
    "a3": "model-local + GLOBAL, but the global half is inert here",
    "a4": "model-local (PAIRFORMER_FLAGS)",
    "a5": "model-local flag + PROCESS-GLOBAL triatt_sdpa._CKC_OVERRIDE (all six models' fused calls)",
    "a6": "model-local flag + PROCESS-GLOBAL triatt_sdpa._CKC_OVERRIDE (all six models' fused calls)",
    "a7": "model-local flag + PROCESS-GLOBAL triatt_sdpa._CKC_OVERRIDE (all six models' fused calls)",
    "a8": "model-local flag + PROCESS-GLOBAL triatt_sdpa._CKC_OVERRIDE (all six models' fused calls)",
    "a9": "model-local flag + PROCESS-GLOBAL triatt_sdpa._CKC_OVERRIDE (all six models' fused calls)",
    "a10": "model-local, RF3-scoped: PAIRFORMER_FLAGS['tri_att_sdpa_hifi'], which reaches RF3's "
           "trunk Pairformer and confidence head and no other model",
}


def apply_arm(name: str) -> dict:
    """Select `name`. Returns what actually changed, resolved from the modules themselves."""
    if name not in ARMS:
        raise SystemExit(f"unknown arm {name!r}; have {sorted(ARMS)}")
    from tt_bio import tenstorrent as tts
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS

    spec = ARMS[name]
    changed = {}
    for k in ("fp32_softmax", "accurate_softmax"):
        if k in spec:
            changed[k] = [PAIRFORMER_FLAGS[k], spec[k]]
            PAIRFORMER_FLAGS[k] = spec[k]
    if spec.get("sdpa_hifi_site"):
        # The per-site selector is read at CONSTRUCTION, so this must land in PAIRFORMER_FLAGS
        # before the model is built -- same rule as fp32_softmax above.
        changed["tri_att_sdpa_hifi"] = [PAIRFORMER_FLAGS.get("tri_att_sdpa_hifi"), True]
        PAIRFORMER_FLAGS["tri_att_sdpa_hifi"] = True
    if "sdpa_ckc" in spec:
        from tt_bio import triatt_sdpa as pm
        changed["_CKC_OVERRIDE"] = [str(pm._CKC_OVERRIDE), spec["sdpa_ckc"]]
        pm._CKC_OVERRIDE = pm.ckc_from_env(spec["sdpa_ckc"])
    if "triatt_fused_hifi" in spec:
        changed["_TRIATT_FUSED_HIFI"] = [bool(tts._TRIATT_FUSED_HIFI),
                                         spec["triatt_fused_hifi"]]
        tts._TRIATT_FUSED_HIFI = spec["triatt_fused_hifi"]

    # A stray BOLTZ2_FP32_SOFTMAX=1 forces the fp32 branch regardless of the model flag, which
    # would silently turn a1 back into a0. Refuse rather than mislabel the row.
    if tts._FP32_SOFTMAX and not spec.get("fp32_softmax", True):
        raise SystemExit("BOLTZ2_FP32_SOFTMAX=1 is set; it overrides fp32_softmax=False and "
                         f"arm {name} would not be the arm. Unset it.")

    return {"arm": name, "route": ROUTE[name], "scope": SCOPE[name], "changed": changed,
            "resolved": {"fp32_softmax": PAIRFORMER_FLAGS["fp32_softmax"],
                         "accurate_softmax": PAIRFORMER_FLAGS["accurate_softmax"],
                         "_TRIATT_FUSED_HIFI": bool(tts._TRIATT_FUSED_HIFI),
                         "_FP32_SOFTMAX": bool(tts._FP32_SOFTMAX),
                         "tri_att_sdpa_hifi": bool(PAIRFORMER_FLAGS.get("tri_att_sdpa_hifi")),
                         "_CKC_OVERRIDE": _ckc_str()}}


def route_counters() -> dict:
    """What the fold ACTUALLY did with the fused kernel, counted rather than argued.

    A compute-config arm only reaches the calls the persistent-mask kernel serves. Where the
    kernel declines, `_tri_att_sdpa_at` falls back to `ttnn.transformer.scaled_dot_product_
    attention`, which takes no compute config at all, so those calls run the op default whatever
    the arm asked for. An arm served on 3 % of calls is dark, not neutral, and the two are
    indistinguishable from the outside without these counts.

    `chunk_picks` is here for a second reason: `fp32_dest_acc` doubles the DST a tile needs, so an
    arm can be pushed off a wide q_chunk onto a narrow one and pay a ROUTE change (1.08-1.81x)
    inside what reads as a fidelity change. If two arms report different picks, the reading bundles
    two effects and has to be split before either is priced.
    """
    from tt_bio import tenstorrent as tts
    from tt_bio import triatt_sdpa as pm
    return {
        "pm_served": pm.STATS[0], "pm_declined": pm.STATS[1],
        "pm_rejects": {"%s %s" % (k[0], list(k[1])): v for k, v in pm.REJECTS.items()},
        "pm_over_l1": [list(k) for k in pm._PM_OVER_L1],
        "pm_l1_errors": {str(list(k)): v[:200] for k, v in pm.PM_L1_ERRORS.items()},
        "sdpa_route_counts": dict(tts.SDPA_ROUTE_COUNTS),
        "sdpa_chunk_picks": {"%dx%d" % k: v for k, v in tts.SDPA_CHUNK_PICKS.items()},
        "sdpa_hifi_calls": tts.SDPA_HIFI_CALLS[0],
        "ckc_resolved": _ckc_str(),
    }


def clear_l1_latches() -> None:
    """Forget every "this config did not fit L1" memo, so an arm is not handed the previous arm's
    retirements.

    Measured on this branch (`perf/rf3/results/hifi_route_probe_768.json`): at 768 aa the op
    default tries q_chunk 768, is refused, and retires it in `_PM_OVER_L1`. Any arm that runs
    LATER in the same process inherits that retirement and is never even offered q768 -- so the
    arm order, not the arm, decides which q_chunk each one gets. Clearing between arms costs one
    refused compile per arm and makes the picks comparable.
    """
    from tt_bio import tenstorrent as tts
    from tt_bio import triatt_sdpa as pm
    pm._PM_OVER_L1.clear()
    pm.PM_L1_ERRORS.clear()
    pm.REJECTS.clear()
    pm.STATS[0] = pm.STATS[1] = 0
    tts._SDPA_Q_CHUNK_OVER_L1.clear()
    tts._SDPA_QK_OVER_L1.clear()
    tts._TRIATT_HIFI_OVER_L1.clear()
    tts.SDPA_ROUTE_COUNTS["fused"] = tts.SDPA_ROUTE_COUNTS["stock"] = 0
    tts.SDPA_CHUNK_PICKS.clear()
    tts.SDPA_HIFI_CALLS[0] = 0


def _ckc_str():
    """The fused kernel's resolved compute config, as it will actually be used. `None` means the op
    default, which is the low-precision `(HiFi2, math_approx, no fp32_dest_acc)` -- naming it here
    keeps a row from being labelled by an arm that did not take."""
    from tt_bio import triatt_sdpa as pm
    c = pm._CKC_OVERRIDE
    if c is None:
        return "op_default(HiFi2,approx,no_acc)"
    return f"{str(c[0]).rsplit('.', 1)[-1]},approx={bool(c[1])},fp32_dest_acc={bool(c[2])}"
