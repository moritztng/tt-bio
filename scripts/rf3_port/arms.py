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
}

ROUTE = {
    "a0": "_fp32_softmax_attention (materialised fp32 softmax, 5-op chain) -- SHIPPED",
    "a1": "_tri_att_sdpa (fused SDPA at the op-default config: HiFi2, math_approx, no fp32_dest_acc)",
    "a2": "_tri_att_sdpa_hifi (fused SDPA at HiFi4 + fp32_dest_acc) above 128 tokens, "
          "_fp32_softmax_attention below",
    "a3": "identical to a1: _TRIATT_FUSED_HIFI is read only inside the fp32_softmax branch",
    "a4": "_fp32_softmax_attention with ttnn.softmax instead of the 5-op reduction chain",
    "a5": "_tri_att_sdpa at HiFi4 + math_approx OFF + fp32_dest_acc ON (a1's route, a1's chunks)",
    "a6": "_tri_att_sdpa at HiFi4 + math_approx ON + fp32_dest_acc ON (a5 minus the approx knob)",
}

SCOPE = {
    "a0": "model-local (shipped default)",
    "a1": "model-local (PAIRFORMER_FLAGS)",
    "a2": "GLOBAL (TT_BIO_TRIATT_FUSED_HIFI on the shared tenstorrent.py)",
    "a3": "model-local + GLOBAL, but the global half is inert here",
    "a4": "model-local (PAIRFORMER_FLAGS)",
    "a5": "model-local flag + PROCESS-GLOBAL triatt_sdpa._CKC_OVERRIDE (all six models' fused calls)",
    "a6": "model-local flag + PROCESS-GLOBAL triatt_sdpa._CKC_OVERRIDE (all six models' fused calls)",
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
                         "_CKC_OVERRIDE": _ckc_str()}}


def _ckc_str():
    """The fused kernel's resolved compute config, as it will actually be used. `None` means the op
    default, which is the low-precision `(HiFi2, math_approx, no fp32_dest_acc)` -- naming it here
    keeps a row from being labelled by an arm that did not take."""
    from tt_bio import triatt_sdpa as pm
    c = pm._CKC_OVERRIDE
    if c is None:
        return "op_default(HiFi2,approx,no_acc)"
    return f"{str(c[0]).rsplit('.', 1)[-1]},approx={bool(c[1])},fp32_dest_acc={bool(c[2])}"
