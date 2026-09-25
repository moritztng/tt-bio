#!/usr/bin/env python3
"""of3t-covdefault: what the flag can and cannot add to the charter's coverage reading.

The charter grades GRADIENTS on `coverage_total.pct_of_model_compared` in
`perf/of3t_modelboundary/MODEL_withtrunk_n384.json`, and the row was dispatched on the
arithmetic 97.98499306866148 + 1.5202384841128946 = 99.50523155277437, i.e. that turning
TT_BIO_OF3_DEVICE_REFATOM on makes all SEVENTEEN host-applied tensors reachable.

It makes EIGHT of them reachable. This measures which eight, and what they are worth:

  * the float64 reference is re-read here and its denominator recomputed, so a disagreement
    with the published 10.279642678524981 would say the invocation is wrong before it says
    anything about the flag;
  * the seventeen are split into the three groups the CODE actually treats differently;
  * the shipped and flag-ON gradient arms of `of3t-hostleg` are diffed by KEY SET, which is
    the direct evidence of what the flag adds to a scored arm rather than an inference from
    the source.

  python3 perf/of3t_covdefault/covmass.py
"""
import hashlib
import json
from pathlib import Path

import torch

W = Path(__file__).resolve().parents[2]
F64 = Path("/home/ttuser/of3t_hostleg/grads_f64_043.pt")
PIN = "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4"
ARM_OFF = Path("/home/ttuser/of3t_hostleg/device_grads_hl_shipped.pt")
ARM_ON = Path("/home/ttuser/of3t_hostleg/device_grads_hl_refatom.pt")
SEVENTEEN = W / "perf/of3t_hostleg/SEVENTEEN.json"
BOUNDARY = W / "perf/of3t_modelboundary/MODEL_withtrunk_n384.json"
READABLE = W / "perf/of3t_readable_mass/READABLE_MASS.json"
BAR = 99.2594
OUT = Path(__file__).with_name("COVERAGE_CEILING.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def mass(ref: dict, names) -> float:
    return float(sum(float(ref[n].double().norm() ** 2) for n in names if n in ref))


def keys_of(p: Path):
    d = torch.load(p, map_location="cpu", mmap=True)
    for k in ("grads", "state_dict"):
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            d = d[k]
            break
    return set(d), d


def main():
    got = sha256(F64)
    ref, _ = keys_of(F64)
    refd = torch.load(F64, map_location="cpu", mmap=True)
    if isinstance(refd, dict) and "grads" in refd and isinstance(refd["grads"], dict):
        refd = refd["grads"]
    refd = {k: v for k, v in refd.items() if v is not None}
    denom = float(sum(float(v.double().norm() ** 2) for v in refd.values()))

    s17 = json.loads(SEVENTEEN.read_text())
    names17 = s17["names"]
    diff8 = [n for n in names17 if n.startswith("diffusion_module.")]
    ie8 = [n for n in names17
           if n.startswith("input_embedder.") and "ref_atom_feature_embedder" in n]
    lq = [n for n in names17 if n.endswith("linear_q.0.weight")]

    off_keys, _ = keys_of(ARM_OFF)
    on_keys, _ = keys_of(ARM_ON)
    added = sorted(on_keys - off_keys)
    # the hostleg arms are keyed inside the diffusion module, so compare on the leaf suffix
    added_full = sorted(n for n in names17
                        if any(n.endswith(a) or a.endswith(n.split(".", 1)[1]) for a in added))

    bd = json.loads(BOUNDARY.read_text())
    shipped = bd["coverage_total"]["pct_of_model_compared"]
    rm = json.loads(READABLE.read_text())

    pct = lambda m: 100.0 * m / denom
    out = {
        "instrument": "of3t-covdefault covmass.py -- what TT_BIO_OF3_DEVICE_REFATOM can add to "
                      "coverage_total.pct_of_model_compared, measured against the pinned "
                      "float64 reference and against the two of3t-hostleg gradient arms",
        "host": "qb1 (tt-quietbox), CPU only",
        "float64_reference": {"path": str(F64), "sha256": got, "matches_pin": got == PIN,
                              "n_tensors_with_a_gradient": len(refd)},
        "denominator": {
            "model_squared_gradient_norm_recomputed": denom,
            "published": 10.279642678524981,
            "rel_diff": abs(denom - 10.279642678524981) / 10.279642678524981},
        "seventeen": {
            "all_17_pct_of_model": pct(mass(refd, names17)),
            "published_pct": s17["denominator"]["seventeen_pct_of_model_mass"],
            "diffusion_module_ref_atom_feature_embedder_8": {
                "names": diff8, "pct_of_model": pct(mass(refd, diff8))},
            "input_embedder_ref_atom_feature_embedder_8": {
                "names": ie8, "pct_of_model": pct(mass(refd, ie8))},
            "input_embedder_linear_q_0_weight": {
                "names": lq, "pct_of_model": pct(mass(refd, lq))}},
        "arms": {
            "shipped": {"path": str(ARM_OFF), "n_keys": len(off_keys)},
            "flag_on": {"path": str(ARM_ON), "n_keys": len(on_keys)},
            "keys_the_flag_adds": added,
            "n_keys_the_flag_adds": len(added),
            "keys_the_flag_removes": sorted(off_keys - on_keys)},
        "coverage": {
            "shipped_measured": shipped,
            "bar": BAR,
            "flag_on_projected": shipped + pct(mass(refd, diff8)),
            "flag_on_projected_vs_bar": shipped + pct(mass(refd, diff8)) - BAR,
            "row_brief_arithmetic": shipped + s17["denominator"]["seventeen_pct_of_model_mass"],
            "why_the_brief_arithmetic_overcounts":
                "it credits all seventeen. The flag moves the diffusion module's eight linears "
                "to the card (worker.py:1559) and the input embedder's aggregation head "
                "(openfold3_host_prep.py:331). The input embedder's own eight ref-atom linears "
                "stay on the host at openfold3_host_prep.py:301 in BOTH arms -- the runtime "
                "census counts ref_atom_embed once with the flag on, not zero times. And no arm "
                "in the union scores the input embedder at all: its section reads "
                "n_compared 0 of 98, so linear_q.0.weight cannot enter the reading either."},
        "uncompared_mass_by_class": {
            k: {"n_tensors": v["n_tensors"], "pct_of_model_mass": v["pct_of_model_mass"]}
            for k, v in rm["classes"].items()
            if k not in ("COMPARED", "READ_ON_ANOTHER_BOUNDARY")},
    }
    cls = out["uncompared_mass_by_class"]
    out["ceilings"] = {
        "uncompared_total_pct": bd["coverage_total"]["pct_of_model_uncompared"],
        "host_applied_pct": cls["HOST_APPLIED"]["pct_of_model_mass"],
        "not_host_applied_but_uncompared_pct": sum(
            v["pct_of_model_mass"] for k, v in cls.items() if k != "HOST_APPLIED"),
        "ceiling_if_only_the_17_blocked": 100 - cls["HOST_APPLIED"]["pct_of_model_mass"],
        "ceiling_reachable_with_the_arms_that_exist_flag_off": shipped,
        "ceiling_reachable_with_the_arms_that_exist_flag_on":
            shipped + pct(mass(refd, diff8)),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("float64_reference", "denominator", "coverage", "ceilings")}, indent=1))
    print("keys the flag adds:", out["arms"]["n_keys_the_flag_adds"], out["arms"]["keys_the_flag_adds"])
    print("seventeen split:", json.dumps(
        {k: v["pct_of_model"] for k, v in out["seventeen"].items()
         if isinstance(v, dict) and "pct_of_model" in v}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
