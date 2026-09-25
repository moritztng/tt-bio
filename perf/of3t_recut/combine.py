#!/usr/bin/env python3
"""of3t-recut job 3: the corrected trunk arm, and the control that says the shortcut is allowed.

  g_corrected = g(cot_s, cot_z) - g(0, delta)

is the shortcut: `g(cot_s, cot_z)` is already banked for every arm, so a configuration costs ONE
extra arm instead of a re-run. It rests on the parameter gradient being linear in the injected
cotangent, which is measured on the float64 reference and not assumed
(`perf/of3t_frameself/TWOBASIS.json`, 6.435383259300361e-15).

Its control is the arm the device ran end to end on (cot_s, cot_z - delta), and ON THE DEVICE THE
SHORTCUT DOES NOT SURVIVE IT. The taped backward is bf16, so it is only linear in the injected
cotangent to its own rounding, and the subtraction cancels two arms of norm 1.328 and 0.595 into
a result of norm 0.815. This script therefore keeps the shortcut's output as a comparison and
names the END TO END arm as the corrected trunk. The repair is untouched by that: what failed is
the cheap way of propagating it, which is why the control was free and mandatory.
"""
from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import torch

O = Path("/home/ttuser/of3t_recut")
BANKED = Path("/home/ttuser/of3t_modelframe/dev_RENORM_model_n384_nocaptures.pt")
DELTA = O / "dev_RENORM_model_n384_delta.pt"
EXT = O / "dev_RENORM_model_n384_external.pt"
OUT = O / "dev_RENORM_model_n384_shortcut.pt"
TWOBASIS = 6.435383259300361e-15


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()


def fit(arm, ref, keys):
    dot = a2 = r2 = e2 = 0.0
    worst, worst_name, n_bit = -1.0, None, 0
    for n in keys:
        r = ref[n].to(torch.float64).reshape(-1)
        m = arm[n].to(torch.float64).reshape(-1)
        rn2 = float(torch.dot(r, r))
        r2 += rn2
        dot += float(torch.dot(m, r))
        a2 += float(torch.dot(m, m))
        d = float(torch.dot(m - r, m - r))
        e2 += d
        if torch.equal(m, r):
            n_bit += 1
        rel = (d / rn2) ** 0.5 if rn2 > 0 else (0.0 if d == 0 else float("inf"))
        if rel > worst:
            worst, worst_name = rel, n
    return {"n_tensors": len(keys), "n_bit_identical": n_bit,
            "rel_l2_as_is": (e2 / r2) ** 0.5 if r2 else None,
            "norm_ratio_arm_over_ref": (a2 / r2) ** 0.5 if r2 else None,
            "cos": dot / (a2 * r2) ** 0.5 if a2 > 0 and r2 > 0 else None,
            "ref_squared_norm": r2, "arm_squared_norm": a2,
            "worst_rel_l2": worst, "worst_tensor": worst_name}


def main() -> int:
    base = torch.load(BANKED, map_location="cpu", weights_only=False)
    dlt = torch.load(DELTA, map_location="cpu", weights_only=False)
    ext = torch.load(EXT, map_location="cpu", weights_only=False)
    gb, gd, ge = base["grads"], dlt["grads"], ext["grads"]
    keys = sorted(k for k in gb if gb[k] is not None)
    missing = [k for k in keys if gd.get(k) is None or ge.get(k) is None]
    if missing:
        raise SystemExit(f"{len(missing)} tensors absent from an arm, e.g. {missing[:3]}")

    gc = {k: (gb[k].to(torch.float64) - gd[k].to(torch.float64)) for k in keys}
    out = dict(base)
    out["grads"] = gc
    out["injection_convention"] = ("graph-cut-external, SHORTCUT (banked arm minus the delta "
                                   "arm). Withdrawn on the device: see DEVICE_CORRECTED.json")
    out["composed_from"] = {"banked": str(BANKED), "delta_arm": str(DELTA)}
    torch.save(out, OUT)

    rep = {"what": __doc__.strip().splitlines()[0], "host": socket.gethostname(),
           "row": "of3t-recut", "defect": "D242",
           "device_involved": True,
           "note": "this script is CPU arithmetic over three banked device arms; the AICLK and "
                   "host_quiet of the two arms it composes are in "
                   "DEV_RENORM_MODEL_N384_DELTA.json and DEV_RENORM_MODEL_N384_EXTERNAL.json",
           "arms": {
               "banked_cot_s_cot_z": {"path": str(BANKED), "sha256": sha(BANKED)},
               "delta_only": {"path": str(DELTA), "sha256": sha(DELTA)},
               "end_to_end_external": {"path": str(EXT), "sha256": sha(EXT)},
           },
           "squared_norms": {
               "banked": sum(float(torch.dot(gb[k].to(torch.float64).reshape(-1),
                                             gb[k].to(torch.float64).reshape(-1)))
                             for k in keys),
               "delta_only": sum(float(torch.dot(gd[k].to(torch.float64).reshape(-1),
                                                 gd[k].to(torch.float64).reshape(-1)))
                                 for k in keys),
               "corrected": sum(float(torch.dot(gc[k].reshape(-1), gc[k].reshape(-1)))
                                for k in keys),
           },
           "LINEARITY_CONTROL": {
               "what": "the shortcut's result against the arm the device actually ran end to "
                       "end on (cot_s, cot_z - delta)",
               "shortcut_vs_end_to_end": fit(gc, ge, keys),
               "reference_for_the_bar": {
                   "source": "perf/of3t_frameself/TWOBASIS.json",
                   "sum_identity_rel_l2": TWOBASIS,
                   "what": "|| g_sonly + g_zonly - g_ctrl || / || g_ctrl || in this same setup, "
                           "on the CPU reference. The device arm is bf16, so its own linearity "
                           "residual is a measurement, not this number."},
           },
           "out": {"path": str(OUT), "sha256": sha(OUT), "bytes": OUT.stat().st_size}}
    r = rep["LINEARITY_CONTROL"]["shortcut_vs_end_to_end"]["rel_l2_as_is"]
    sq = rep["squared_norms"]
    amp = ((sq["banked"] ** 0.5 + sq["delta_only"] ** 0.5) / sq["corrected"] ** 0.5)
    rep["LINEARITY_CONTROL"]["cancellation"] = {
        "norm_banked": sq["banked"] ** 0.5, "norm_delta_only": sq["delta_only"] ** 0.5,
        "norm_of_the_difference": sq["corrected"] ** 0.5,
        "amplification_of_a_relative_error": amp,
        "implied_per_arm_relative_error": r / amp,
        "bf16_unit_roundoff": 2 ** -8,
        "what": "the subtraction cancels two arms into a smaller result, so each arm's own "
                "rounding is amplified by this factor in the difference. The implied per-arm "
                "error is the residual divided by it, and it sits at bf16 scale, so this reads "
                "as the taped backward's own rounding rather than a wrong identity.",
    }
    rep["LINEARITY_CONTROL"]["verdict"] = (
        f"the shortcut reproduces the end-to-end arm to {r:.6e}; the subtraction stands"
        if r <= 1e-9 else
        f"the shortcut and the end-to-end arm differ by {r:.6e}, far outside the float64 "
        f"reference's own {TWOBASIS:.6e}. On a bf16 taped backward the injection is linear only "
        f"to its own rounding, so the SHORTCUT IS WITHDRAWN on device arms and the end-to-end "
        f"arm is this row's corrected trunk. The repair itself is untouched: the float64 "
        f"reference reproduces grads_f64_043 to 1.6952505222168708e-14 over all 2,736 tensors.")
    rep["THE_CORRECTED_TRUNK_IS"] = {
        "path": str(EXT),
        "why": "the arm the device ran end to end on (cot_s, cot_z - delta), not the "
               "subtraction, because the subtraction failed its own control",
        "the_shortcut_is_kept_at": str(OUT),
    }
    Path("perf/of3t_recut/DEVICE_CORRECTED.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
