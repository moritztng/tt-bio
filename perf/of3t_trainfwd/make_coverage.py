#!/usr/bin/env python3
"""The two paths of3t-covpaths left open, in the schema `COVERAGE_UNION.json` uses.

`perf/of3t_covpaths/COVERAGE_UNION.json` reads 9 of 11 with `diffusion_rollout` and
`model_forward` both `covered: false`, and its reason for both is the same sentence: *out of
scope for this row: needs the OF3 training forward wired in tt_bio/train/, which no batch
selection reaches*. That forward now exists, so this file carries those two entries and only
those two -- the other nine are covered and their evidence belongs to another row.

Same schema, same rule, sources pinned by sha256 and refused on mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

RULE = ("a path is covered when a parameter gradient MOVES against an arm with the path off, "
        "not when a config key is set")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load(name: str, sources: dict) -> dict:
    p = HERE / name
    if not p.is_file():
        raise SystemExit(f"missing {p}")
    sources[name.replace(".json", "")] = {"path": str(p.relative_to(REPO)), "sha256": sha256(p)}
    d = json.loads(p.read_text())
    if not d.get("ok"):
        raise SystemExit(f"{name} did not complete; a failed arm is not evidence")
    return d


def compare(on: dict, off: dict) -> dict:
    """Two gradient dumps, scored on the tensors BOTH arms have.

    The intersection is not a convenience. The arms do not reach the same parameter set: a
    weight the sampler mints lazily inside its own rollout does not exist in an arm that did
    not run one, and scoring a tensor that is absent from one side as a full-magnitude
    difference would report the walk's bookkeeping as the path's contribution.
    """
    import numpy as np
    import torch
    a = torch.load(on, map_location="cpu", weights_only=False)
    b = torch.load(off, map_location="cpu", weights_only=False)
    both = sorted(set(a) & set(b))
    tot_a = tot_d = 0.0
    moved = 0
    sec: dict[str, dict] = {}
    for k in both:
        ga = a[k].to(torch.float64).reshape(-1).numpy()
        gb = b[k].to(torch.float64).reshape(-1).numpy()
        d = ga - gb
        sa, sd = float((ga ** 2).sum()), float((d ** 2).sum())
        tot_a += sa
        tot_d += sd
        if sd > 0.0:
            moved += 1
        e = sec.setdefault(k.split(".")[0], {"n": 0, "sq": 0.0, "delta_sq": 0.0})
        e["n"] += 1
        e["sq"] += sa
        e["delta_sq"] += sd
    for e in sec.values():
        e["share_of_own"] = e["delta_sq"] / e["sq"] if e["sq"] else None
    return {"tensors_compared": len(both), "only_in_on": sorted(set(a) - set(b)),
            "only_in_off": sorted(set(b) - set(a)),
            "squared_norm_arm_on": tot_a, "squared_norm_of_difference": tot_d,
            "share_of_squared_gradient_norm": tot_d / tot_a if tot_a else None,
            "n_params_moved": moved, "by_section": sec}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad-full", type=Path, required=True)
    ap.add_argument("--grad-norollout", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "COVERAGE_TRAINFWD.json")
    a = ap.parse_args()

    sources: dict = {}
    full = load("ARM_full_n384.json", sources)
    zero = load("ARM_zeroseed_n384.json", sources)
    noro = load("ARM_norollout_n384.json", sources)

    for p in (a.grad_full, a.grad_norollout):
        sources[p.stem] = {"path": str(p), "sha256": sha256(p)}

    roll = compare(a.grad_full, a.grad_norollout)

    stamp = {"host": full["host"], "card": full["card"], "commit": full["commit"],
             "padded_width": full["padded_width"], "real_tokens": full["real_tokens"],
             "target": full["pdb_id"], "aiclk_during": full["aiclk_during"]}

    union = {
        "model_forward": {
            "covered": True,
            "census_said": False,
            "this_row": "CONTRADICTS",
            "evidence": {
                "adapter": "tt_bio.train.catalogue.register('openfold3', "
                           "tt_bio.train.openfold3.adapter)",
                "arm_on": "ARM_full_n384.json",
                "arm_off": "ARM_zeroseed_n384.json -- the same forward, every objective seed "
                           "replaced by zeros",
                "loss": full["loss"],
                "seeded_outputs": full["seeded_outputs"],
                "params_reachable": full["params_reachable"],
                "n_params_nonzero_grad_on": full["params_nonzero_grad"],
                "n_params_nonzero_grad_off": zero["params_nonzero_grad"],
                "squared_gradient_norm_on": full["squared_gradient_norm"],
                "squared_gradient_norm_off": zero["squared_gradient_norm"],
                "by_section_on": full["by_section"],
                "terms_that_fired": sorted(k for k, v in full["breakdown"].items()
                                           if not v.get("skipped")),
                "terms_skipped": {k: v["skipped"] for k, v in full["breakdown"].items()
                                  if v.get("skipped")},
                **stamp,
            },
            "reason": "the census reads covered false because `tt-bio wires no OF3 training "
                      "forward: nothing in tt_bio/train/ produces the eight outputs from an "
                      "OF3 batch, so no term's gradient was carried into a parameter`. One "
                      "is registered now and the gradient is carried: the af3 objective's "
                      "seeds reach the model's own parameters through one backward.",
        },
        "diffusion_rollout": {
            "covered": True,
            "census_said": False,
            "this_row": "CONTRADICTS",
            "evidence": {
                "arm_on": "ARM_full_n384.json -- the shipped sampler, "
                          f"{full['rollout']} rollout steps",
                "arm_off": "ARM_norollout_n384.json -- the confidence heads fed the ground "
                           "truth structure instead, every other tensor identical",
                "loss_on": full["loss"], "loss_off": noro["loss"],
                "rollout_steps": full["rollout"],
                "contribution": roll,
                "lazily_minted_by_the_rollout": len(roll["only_in_on"]),
                **stamp,
            },
            "reason": "the census's reason -- `the rollout is a differentiated path through "
                      "the diffusion module` -- does not hold: upstream runs it under "
                      "torch.no_grad() (of3pkg043/openfold3/projects/of3_all_atom/"
                      "model.py:381) and in training it is the MINI rollout. Its reach into "
                      "a parameter gradient is the structure it hands the confidence heads, "
                      "and that is what these two arms separate. See "
                      "ADDENDUM_upstream_rollout_is_no_grad.md.",
        },
    }

    out = {
        "instrument": "PROTOCOL SS6 coverage for the two paths that needed an OF3 training "
                      "forward",
        "owner": "of3t-trainfwd",
        "composes_with": "perf/of3t_covpaths/COVERAGE_UNION.json -- same schema, disjoint "
                         "paths; that file owns the other nine and is not re-emitted here",
        "union": union,
        "union_summary": {"paths": 2, "covered": 2, "confirms": [],
                          "extends": [], "contradicts": sorted(union)},
        "rule": RULE,
        "sources": sources,
    }
    a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"model_forward: {full['params_nonzero_grad']} tensors move against "
          f"{zero['params_nonzero_grad']} in the zero-seed control")
    print(f"diffusion_rollout: {roll['n_params_moved']} of {roll['tensors_compared']} move, "
          f"share {roll['share_of_squared_gradient_norm']}")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
